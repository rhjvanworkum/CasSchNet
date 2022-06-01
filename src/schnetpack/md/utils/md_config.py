import copy
import logging
from omegaconf import DictConfig, open_dict, OmegaConf
from schnetpack.utils import str2class

from typing import List, Tuple

log = logging.getLogger(__name__)

integrator_to_npt = {
    "schnetpack.md.integrators.VelocityVerlet": "schnetpack.md.integrators.NPTVelocityVerlet",
    "schnetpack.md.integrators.RingPolymer": "schnetpack.md.integrators.NPTRingPolymer",
}


class MDConfigMerger:
    """
    Custom merge tool for two hydra configs in order to circumvent the refusal of hydra to do as wanted. Takes two
    configs, updates the base config obtained from hydra using a second one (e.g. loaded from a file) at the hightest
    config level and then re-applies the hydra overrides.
    """

    def merge_configs(
        self,
        base_config: DictConfig,
        overwrite_config: DictConfig,
        overrides: List[str],
    ) -> DictConfig:
        """
        Combine two configs and re-apply overrides.

        Args:
            base_config (DictConfig): basic config generated by hydra at start of a run.
            overwrite_config (DictConfig): config used for updating base config
            overrides (list(str)): overrides provided by hydra.

        Returns:
            DictConfig: basic config updated with entries from the overwrite config and all overrides re-applied.
        """

        # merge overwrite config into basic config
        merged_config = OmegaConf.merge(base_config, overwrite_config)

        # get overrides in list of lists format
        overrides_modify, overrides_delete = self._parse_overrides(overrides)

        # re-apply overrides to new config
        merged_config = self._override_config_entries(
            merged_config, base_config, overrides_modify
        )
        # delete config entries if requested in overrides
        merged_config = self._delete_config_entries(merged_config, overrides_delete)

        return merged_config

    @staticmethod
    def _override_config_entries(
        merge_config: DictConfig, base_config: DictConfig, overrides: List[List[str]]
    ) -> DictConfig:
        """
        Re-apply overrides from base configs to config updated with the loaded config file. Traverses the list of lists
        created from the overrides and updates the config leaf nodes. This only applies to non-deletion overrides.
        For deletions refer to the `_delete_config_entries` function.

        Args:
            merge_config (DictConfig): updated config.
            base_config (DictConfig): unmodified base config.
            overrides (list(list(str)): overrides.

        Returns:
            DictConfig: updated config with the overrides reapplied.
        """

        for ov in overrides:
            # get top-level override key
            ov_key = ov.pop(0)
            # check if traversal is already finished
            if len(ov) == 0:
                # use base config entry containing parsed override in new config
                merge_config[ov_key] = base_config[ov_key]
            else:
                # if traversal is not finished, use the list of config keys to traverse both configs and apply
                # the overrides from the base config once the lowest level is reached
                current_base = base_config[ov_key]
                current_merge = merge_config[ov_key]
                while True:
                    ov_key = ov.pop(0)
                    if len(ov) == 0:
                        current_merge[ov_key] = current_base[ov_key]
                        break
                    else:
                        current_base = current_base[ov_key]
                        current_merge = current_merge[ov_key]

        return merge_config

    @staticmethod
    def _delete_config_entries(
        merge_config: DictConfig, overrides: List[List[str]]
    ) -> DictConfig:
        """
        Delete config entries given in the  overrides. Traverses the list of lists created from the overrides and
        deletes the corresponding config leaf nodes.

        Args:
            merge_config (DictConfig): updated config.
            overrides (list(list(str)): overrides for deletion.

        Returns:
            DictConfig: updated config with the overrides deleted.
        """

        # use open_dict context to allow deletion of keys
        with open_dict(merge_config):
            for ov in overrides:
                # get top-level override key
                ov_key = ov.pop(0)
                # check if traversal is already finished
                if len(ov) == 0:
                    # delete entry merge config
                    if ov_key in merge_config:
                        del merge_config[ov_key]
                else:
                    # if traversal is not finished, use the list of config keys to traverse config and delete
                    # the entry once the lowest level is reached
                    current_merge = merge_config[ov_key]
                    while True:
                        ov_key = ov.pop(0)
                        if len(ov) == 0:
                            if ov_key in current_merge:
                                del current_merge[ov_key]
                            break
                        else:
                            current_merge = current_merge[ov_key]

        return merge_config

    def _parse_overrides(
        self, overrides: List[str]
    ) -> Tuple[List[List[str]], List[List[str]]]:
        """
        Parse hydra override entries, split them into modification and deletion events and then convert them into lists
        of lists of config keys which can be used for traversing the configs.

        Args:
            overrides (list(str)):

        Returns:
            (list(list(str)), list(list(str))): override specifications for modifying and deleting config entries.
        """
        overrides_modify = []
        overrides_delete = []

        # split overrides into modification +/'' or deletion ~ statements
        for override in overrides:
            if override.startswith("~"):
                overrides_delete.append(override)
            else:
                overrides_modify.append(override)

        # convert overrides to lost of list format of config paths
        overrides_modify = self._convert_overrides(overrides_modify)
        overrides_delete = self._convert_overrides(overrides_delete)

        return overrides_modify, overrides_delete

    @staticmethod
    def _convert_overrides(overrides: List[str]) -> List[List[str]]:
        """
        Convert raw overrides into list of lists, removing override characters and using `/` and '.' to determine
        their structure.

        Args:
            overrides (list(str)): list of override statements from hydra.

        Returns:
            list(list(str)): List of lists, where each list corresponds to a full path in the config file.
        """
        # remove +/~ from override statements and get basic names
        split_overrides = [ov.strip("+~").rsplit("=", 1)[0] for ov in overrides]
        # split override statements at '/' and '.'
        split_overrides = [ov.replace("/", ".").split(".") for ov in split_overrides]
        return split_overrides


def is_rpmd_integrator(integrator_type: str):
    """
    Check if an integrator is suitable for ring polymer molecular dynamics.

    Args:
        integrator_type (str): integrator class name

    Returns:
        bool: True if integrator is suitable, False otherwise.
    """
    integrator_class = str2class(integrator_type)

    if hasattr(integrator_class, "ring_polymer"):
        return integrator_class.ring_polymer
    else:
        log.warning(
            "Could not determine if integrator is suitable for ring polymer simulations."
        )
        return False


def get_npt_integrator(integrator_type: str):
    """
    Check if integrator is suitable for constant pressure dynamics and determine the constant pressure equivalent if
    this is not the case.

    Args:
        integrator_type (str): name of the integrator class.

    Returns:
        str: class of suitable constant pressure integrator.
    """
    integrator_class = str2class(integrator_type)

    if hasattr(integrator_class, "pressure_control"):
        if integrator_class.pressure_control:
            return integrator_type
        else:
            # Look for constant pressure equivalent
            if integrator_type in integrator_to_npt:
                log.info(
                    "Switching integrator from {:s} to {:s} for constant pressure simulation...".format(
                        integrator_type, integrator_to_npt[integrator_type]
                    )
                )
                return integrator_to_npt[integrator_type]
                # If NPT suitability can not be determined automatically, good luck
            else:
                log.warning(
                    "No constant pressure equivalent for integrator {:s} could be found.".format(
                        integrator_type
                    )
                )
            return integrator_type
    else:
        log.warning(
            "Please check whether integrator {:s} is suitable for constant pressure"
            " simulations (`pressure control` attribute).".format(integrator_type)
        )
        return integrator_type
