import math
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
from scipy.stats import genpareto, lognorm
from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field, validator

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Internal Pydantic validation models
# ---------------------------------------------------------------------------

class component_data_model(BaseModel):
    """Represents a single row of component inventory data.

    Attributes
    ----------
    Component_ID : Optional[int]
        Unique identifier for the component. Must be a positive integer.
    Description : Optional[str]
        Textual description of the component.
    EDP : str
        Engineering Demand Parameter (e.g., ``'psd'``, ``'pfa'``).
    Typology : str
        Component type (e.g., ``'s'`` structural, ``'ns'`` non-structural).
    Performance_Group : Optional[int]
        Classification group number.
    Quantity : float
        Number of units of this component present on the storey.
    Damage_States : int
        Number of defined damage states.
    """

    Component_ID:      Optional[int] = Field(alias="Component ID")
    Description:       Optional[str] = None
    EDP:                        str
    Typology:                   str
    Performance_Group: Optional[int] = Field(alias="Performance Group")
    Quantity:                  float
    Damage_States:               int = Field(alias="Damage States")

    @validator("Component_ID")
    def validate_id(cls, v):
        if v is not None and v < 0:
            raise ValueError("Component ID must be a positive integer")
        return v

    @validator("Performance_Group", "Component_ID", pre=True)
    def allow_none(cls, v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        return v


class correlation_tree_model(BaseModel):
    """Tracks the dependency structure between components.

    Attributes
    ----------
    ID : int
        Unique identifier for the correlation entry (must be positive).
    dependent_on_item : str
        Name of the item this component depends on.
    """

    ID: int
    dependent_on_item: str = Field(alias="DEPENDENT ON ITEM")

    @validator("ID")
    def validate_id(cls, vid):
        if vid < 0:
            raise ValueError("Component ID must be a positive integer")
        return vid


class item_base(BaseModel):
    """Stores a dictionary of arrays for a single item.

    Attributes
    ----------
    RootModel : Dict[str, np.ndarray]
        Mapping of label → array.
    """

    RootModel: Dict[str, np.ndarray]

    class Config:
        arbitrary_types_allowed = True


class items_model(BaseModel):
    """Collection of items keyed by integer ID.

    Attributes
    ----------
    RootModel : Dict[int, item_base]
        Mapping of item ID → :class:`item_base`.
    """

    RootModel: Dict[int, item_base]


class fragility_model(BaseModel):
    """Holds EDP values and the associated item fragility data.

    Attributes
    ----------
    EDP : np.ndarray
        Engineering Demand Parameter values.
    ITEMs : items_model
        Fragility data for all items.
    """

    EDP: np.ndarray
    ITEMs: items_model

    class Config:
        arbitrary_types_allowed = True


class ds_model(BaseModel):
    """Damage-state arrays nested by item ID and simulation index.

    Attributes
    ----------
    RootModel : Dict[int, Dict[int, np.ndarray]]
        ``{item_id: {simulation_index: damage_state_array}}``.
    """

    RootModel: Dict[int, Dict[int, np.ndarray]]

    class Config:
        arbitrary_types_allowed = True


class cost_model(BaseModel):
    """Cost arrays keyed by component ID.

    Attributes
    ----------
    RootModel : Dict[int, np.ndarray]
        Mapping of component ID → cost array.
    """

    RootModel: Dict[int, np.ndarray]

    class Config:
        arbitrary_types_allowed = True


class simulation_model(BaseModel):
    """Per-simulation repair costs keyed by component ID.

    Attributes
    ----------
    RootModel : Dict[int, cost_model]
        Mapping of component ID → :class:`cost_model`.
    """

    RootModel: Dict[int, cost_model]

    class Config:
        arbitrary_types_allowed = True


class fitting_model(BaseModel):
    """Optimised parameters and covariance matrix from curve fitting.

    Attributes
    ----------
    popt : np.ndarray
        Optimised parameter vector.
    pcov : np.ndarray
        Covariance matrix of the optimised parameters.
    """

    popt: np.ndarray
    pcov: np.ndarray

    class Config:
        arbitrary_types_allowed = True


class fitting_parameters_model(BaseModel):
    """Fitting results for multiple quantiles.

    Attributes
    ----------
    RootModel : Dict[str, fitting_model]
        Mapping of quantile label → :class:`fitting_model`.
    """

    RootModel: Dict[str, fitting_model]


class fitted_loss_model(BaseModel):
    """Fitted loss-function arrays for multiple quantiles.

    Attributes
    ----------
    RootModel : Dict[str, np.ndarray]
        Mapping of quantile label → fitted loss array.
    """

    RootModel: Dict[str, np.ndarray]

    class Config:
        arbitrary_types_allowed = True


class loss_model(BaseModel):
    """Absolute and normalised loss results per component.

    Attributes
    ----------
    loss : Dict[int, Dict[Union[int, str], float]]
        Absolute loss values keyed by component ID and damage state.
    loss_ratio : Dict[int, Dict[Union[int, str], float]]
        Normalised loss ratios keyed by component ID and damage state.
    """

    loss: Dict[int, Dict[Union[int, str], float]]
    loss_ratio: Dict[int, Dict[Union[int, str], float]]


class slf_model(BaseModel):
    """Storey Loss Function output record.

    Attributes
    ----------
    directionality : Optional[int]
        Analysis directionality flag.
    component_type : str
        Component type label (e.g., ``'PSD, NS'``).
    storey : Optional[Union[int, List[int]]]
        Storey level(s) covered by this SLF.
    edp : str
        Engineering Demand Parameter label.
    edp_range : List[float]
        EDP values over which the SLF is defined.
    slf : List[float]
        Mean Storey Loss Function values.
    """

    directionality: Optional[int]  = Field(alias="Directionality")
    component_type: str             = Field(alias="Component-type")
    storey: Optional[Union[int, List[int]]] = Field(alias="Storey")
    edp: str
    edp_range: List[float]
    slf: List[float]


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class slfgenerator:
    """Storey Loss Function (SLF) generator for storey-based loss assessment.

    Automates the generation of Storey Loss Functions using fragility,
    consequence, and quantity data via a probabilistic Monte-Carlo approach.

    References
    ----------
    1. Ramirez and Miranda (2009). *Building-Specific Loss Estimation Methods
       & Tools for Simplified PBEE*. John A. Blume EEC, Stanford University.
    2. Shahnazaryan D, O'Reilly GJ, Monteiro R. (2021). Story loss functions
       for seismic design and assessment. *Earthquake Spectra*, 37(4):
       2813–2839. https://doi.org/10.1177/87552930211023523
    3. Shahnazaryan D, O'Reilly GJ, Monteiro R. (2021). Development of a
       Python-Based Storey Loss Function Generator. *COMPDYN 2021*.
       https://doi.org/10.7712/120121.8659.18567

    Acknowledgements
    ----------------
    Based on the original work by Dr. Davit Shahnazaryan:
    https://github.com/davitshahnazaryan3/SLFGenerator
    """

    def __init__(self,
                 component_data: component_data_model,
                 edp: str,
                 correlation_tree: correlation_tree_model = None,
                 typology: List[str] = None,
                 edp_range: Union[List[float], np.ndarray] = None,
                 edp_bin: float = None,
                 grouping_flag: bool = True,
                 conversion: float = 1.0,
                 realizations: int = 20,
                 replacement_cost: float = 1.0,
                 regression: str = "Weibull",
                 storey: Union[int, List[int]] = None,
                 directionality: int = None):
        """Initialise the SLF generator.

        Parameters
        ----------
        component_data : pandas.DataFrame
            Inventory of component data (loaded from CSV).
        edp : str
            Engineering Demand Parameter; ``'PSD'`` (Peak Storey Drift) or
            ``'PFA'`` (Peak Floor Acceleration).
        correlation_tree : pandas.DataFrame, optional
            Correlation tree defining component dependencies. Default ``None``.
        typology : List[str], optional
            Component typologies to include (``'ns'`` or ``'s'``).
            Default ``None``.
        edp_range : array-like, optional
            Custom EDP value range. If ``None``, defaults are used.
        edp_bin : float, optional
            EDP bin size. If ``None``, a type-specific default is used.
        grouping_flag : bool, optional
            Whether to group components by performance group. Default ``True``.
        conversion : float, optional
            Cost conversion factor. Default ``1.0``.
        realizations : int, optional
            Number of Monte Carlo realizations. Default ``20``.
        replacement_cost : float, optional
            Normalising replacement cost. Default ``1.0``.
        regression : str, optional
            Regression model: ``'Weibull'``, ``'Papadopoulos'``, ``'Gdp'``,
            or ``'Lognormal'``. Default ``'Weibull'``.
        storey : int or List[int], optional
            Storey level(s) to include. Default ``None``.
        directionality : int, optional
            Analysis directionality flag. Default ``None``.
        """
        self.edp = edp.lower()
        self.typology = typology
        self.edp_bin = edp_bin
        self.edp_range = edp_range
        self.grouping_flag = grouping_flag
        self.conversion = conversion
        self.realizations = realizations
        self.replacement_cost = replacement_cost
        self.regression = regression.lower() if regression is not None else None
        self.storey = storey
        self.directionality = directionality
        self.correlation_tree = correlation_tree

        # Normalise all string entries to lowercase
        self.component_data = component_data.map(
            lambda s: s.lower() if isinstance(s, str) else s
        )

        # Set up EDP range and parse component inventory
        self._define_edp_range()
        self._get_component_data()

        # Process optional correlation tree
        if self.correlation_tree is not None:
            self.correlation_tree = self.correlation_tree.map(
                lambda s: s.lower() if isinstance(s, str) else s
            )
            self._get_correlation_tree()

        # Group components by performance group if requested
        if self.grouping_flag:
            self._group_components()

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _define_edp_range(self):
        """Set up the EDP discretisation range.

        Raises
        ------
        ValueError
            If ``edp`` is not ``'psd'``, ``'idr'``, or ``'pfa'``.
        """
        edp_defaults = {
            "idr": (0.1 / 100, 0, 0.5),
            "psd": (0.1 / 100, 0, 0.5),
            "pfa": (0.05,      0, 5.0),
        }

        if self.edp not in edp_defaults:
            raise ValueError(
                "Incorrect EDP provided — must be 'psd', 'idr', or 'pfa'."
            )

        default_bin, range_start, range_end = edp_defaults[self.edp]
        if self.edp_bin is None:
            self.edp_bin = default_bin

        if self.edp_range is None:
            self.edp_range = np.arange(
                range_start, range_end + self.edp_bin, self.edp_bin
            )

        self.edp_range = np.asarray(self.edp_range, dtype=float)
        self.edp_range[0] = 1e-20

    def _get_component_data(self):
        """Parse and validate the component inventory DataFrame.

        Missing ``Component ID`` values are filled with sequential integers;
        missing ``Description`` values default to ``'B'``; missing values in
        all other columns (except ``Performance Group`` and ``Typology``) are
        set to ``0``.
        """
        self._validate_component_data_schema()

        # Fill missing 'Best Fit' columns with 'normal'
        best_fit_cols = [
            col for col in self.component_data if col.endswith("Best Fit")
        ]
        self.component_data[best_fit_cols] = (
            self.component_data[best_fit_cols].fillna("normal")
        )

        # Auto-assign missing component IDs
        self.component_data["Component ID"] = (
            self.component_data["Component ID"].fillna(
                pd.Series(
                    np.arange(1, len(self.component_data) + 1), dtype="int"
                )
            )
        )
        self.component_data["Description"] = (
            self.component_data["Description"].fillna("B")
        )

        # Fill remaining columns (excluding categorical ones) with 0
        exclude_cols = ["Performance Group", "Typology"]
        cols_to_fill = self.component_data.columns.difference(exclude_cols)
        self.component_data[cols_to_fill] = (
            self.component_data[cols_to_fill].fillna(0)
        )

    def _group_components(self):
        """Partition components into performance groups.

        Groups are formed by ``(EDP, Typology)`` pairs, giving three standard
        buckets: ``'PSD, S'``, ``'PSD, NS'``, and ``'PFA, NS'``.  If explicit
        ``Performance Group`` values are present they override this default
        grouping.
        """
        self.component_data["Performance Group"] = self.component_data["Performance Group"].fillna(-1)
        self.component_data["Typology"] = self.component_data["Typology"].fillna("-1")

        if not self.grouping_flag:
            key = self.component_data["EDP"].iloc[0]
            self.component_groups = {key: self.component_data}
            return

        edp_groups = self.component_data.groupby(["EDP", "Typology"])

        psd_s = (
            edp_groups.get_group(("psd", "s"))
            if ("psd", "s") in edp_groups.groups else None
        )
        psd_ns = (
            edp_groups.get_group(("psd", "ns"))
            if ("psd", "ns") in edp_groups.groups else None
        )
        pfa_ns = (
            edp_groups.get_group(("pfa", "ns"))
            if ("pfa", "ns") in edp_groups.groups else None
        )

        self.component_groups = {
            k: v for k, v in {
                "PSD, S":  psd_s,
                "PSD, NS": psd_ns,
                "PFA, NS": pfa_ns,
            }.items() if v is not None
        }

        # Override with explicit performance groups if more than one exists
        if self.component_data["Performance Group"].nunique() > 1:
            self.component_groups = {
                group: df
                for group, df in self.component_data.groupby("Performance Group")
            }

    def _get_correlation_tree(self):
        """Build the internal correlation matrix from the correlation tree.

        The matrix has shape
        ``(n_components, n_damage_states + 1)`` where the first column stores
        the causation component ID and the remaining columns store the minimum
        damage state required on the causation component before the dependent
        component sustains damage.

        Notes
        -----
        The user is responsible for ensuring that the minimum DS assigned does
        not exceed the available DS defined for each component.  This is not
        validated automatically.
        """
        damage_states = list(self.component_data["Damage States"])
        correlation_data = self.correlation_tree.loc[
            self.component_data.index
        ].values

        self._validate_correlation_tree_schema(damage_states)

        item_ids = correlation_data[:, 0]
        correlation_values = np.delete(correlation_data, 0, axis=1)
        self.matrix = np.full(correlation_values.shape, np.nan, dtype=float)

        for i, row in enumerate(correlation_values):
            for j, value in enumerate(row):
                if j == 0:
                    if isinstance(value, str) and value.lower() == "independent":
                        self.matrix[i, j] = item_ids[i]
                    elif not item_ids[i] or math.isnan(item_ids[i]):
                        self.matrix[i, j] = np.nan
                    else:
                        self.matrix[i, j] = value
                else:
                    if math.isnan(self.matrix[i, j - 1]):
                        self.matrix[i, j] = np.nan
                    elif isinstance(value, str) and value.lower() in {
                        "independent", "undamaged"
                    }:
                        self.matrix[i, j] = 0
                    else:
                        self.matrix[i, j] = int(value[-1])

    def _validate_component_data_schema(self):
        """Validate required columns and their counts in the component data.

        Raises
        ------
        ValueError
            If duplicate ``Component ID`` values are found, or if the counts of
            ``Median``, ``Total Dispersion``, ``Cost``, ``Cost Dispersion``,
            and ``Best Fit`` columns are not equal.
        """
        columns = list(self.component_data.columns)
        component_data = self.component_data.to_dict(orient="records")

        id_set = set()
        for row in component_data:
            model = component_data_model.model_validate(row)
            if model.Component_ID is not None and model.Component_ID in id_set:
                raise ValueError(f"Duplicate Component ID: {model.Component_ID}")
            id_set.add(model.Component_ID)

        counts = {
            "Median": 0,
            "Total Dispersion": 0,
            "Cost": 0,
            "Cost Dispersion": 0,
            "Best Fit": 0,
        }
        for col in columns:
            for key in counts:
                if col.endswith(key):
                    counts[key] += 1

        expected = counts["Median"]
        for key, count in counts.items():
            if count != expected:
                raise ValueError(
                    "Column counts must be equal for 'Median', "
                    "'Total Dispersion', 'Cost', 'Cost Dispersion', "
                    "and 'Best Fit'."
                )

    def _validate_correlation_tree_schema(self, damage_states):
        """Validate the correlation tree against the component inventory.

        Parameters
        ----------
        damage_states : list of int
            Number of damage states for each component (in order).

        Raises
        ------
        ValueError
            On duplicate IDs, insufficient columns, DS range violations, or
            dimension mismatches between the tree and the component data.
        """
        corr_dict = self.correlation_tree.to_dict(orient="records")

        id_set = set()
        for row in corr_dict:
            model = correlation_tree_model.model_validate(row)
            if model.ID in id_set:
                raise ValueError(f"Duplicate ITEM: {model.ID}")
            id_set.add(model.ID)

        if len(self.correlation_tree.keys()) < max(damage_states) + 3:
            raise ValueError(
                "Unexpected (fewer) number of features in the correlations "
                "DataFrame."
            )

        for idx, item in enumerate(self.component_data.index):
            for feature in self.correlation_tree.keys():
                ds = str(self.correlation_tree.loc[item][feature])
                if ds == f"DS{damage_states[idx] + 1}":
                    raise ValueError(
                        "MIN DS in the correlation tree must not exceed the "
                        "possible DS defined for the element."
                    )

        if len(self.component_data) != len(self.correlation_tree):
            raise ValueError(
                "Number of items in the correlation tree and component data "
                "must match."
            )

    def _fit_regression(self, losses, edp_range, fitting_function, percentiles):
        """Fit a single regression model across all requested quantiles.

        Parameters
        ----------
        losses : dict
            Quantile tables for ``'loss'`` and ``'loss_ratio'``.
        edp_range : np.ndarray
            EDP values used as the independent variable for fitting.
        fitting_function : dict
            Dictionary with keys ``'func'`` (callable) and ``'p0'`` (initial
            parameter guess list).
        percentiles : list of float
            Quantiles to fit (e.g. ``[0.16, 0.50, 0.84]``).

        Returns
        -------
        losses_fitted : dict
            Fitted loss-ratio arrays keyed by quantile label and ``'mean'``.
        fitting_parameters : dict
            Fitted ``popt`` and ``pcov`` keyed by quantile label and ``'mean'``.
        error_max : float
            Maximum absolute regression error as a percentage.
        error_cum : float
            Cumulative regression error as a percentage.
        """
        losses_fitted = {}
        fitting_parameters = {}

        for q in percentiles + ["mean"]:
            max_val = max(losses["loss_ratio"].loc[q])
            normalised = losses["loss_ratio"].loc[q] / max_val

            try:
                popt, pcov = curve_fit(
                    fitting_function["func"],
                    edp_range,
                    normalised,
                    p0=fitting_function["p0"],
                    maxfev=10 ** 6,
                )
            except RuntimeError as exc:
                print(
                    f"Regression failed at quantile {q}: {exc}"
                )
                continue

            fitted = fitting_function["func"](edp_range, *popt) * max_val
            fitted[fitted <= 0] = 0.0
            losses_fitted[q] = fitted
            fitting_parameters[q] = {"popt": popt, "pcov": pcov}

        error_max, error_cum = self.estimate_accuracy(
            losses["loss_ratio"].loc["mean"], losses_fitted["mean"]
        )
        return losses_fitted, fitting_parameters, error_max, error_cum

    # -----------------------------------------------------------------------
    # Public methods
    # -----------------------------------------------------------------------

    def fragility_function(self) -> tuple:
        """Derive lognormal fragility functions for all components.

        Returns
        -------
        fragilities : dict
            Keys ``'EDP'`` (np.ndarray) and ``'IDs'`` (nested dict mapping
            component index → DS label → exceedance probability array).
        means_cost : np.ndarray
            Shape ``(n_components, n_ds)`` — mean repair costs per DS.
        covs_cost : np.ndarray
            Shape ``(n_components, n_ds)`` — cost CoV per DS.
        """
        n_ds = self.component_data.columns.str.endswith("Median").sum()

        data = self.component_data.select_dtypes(exclude=["object"]).drop(
            labels=["Component ID", "Performance Group", "Quantity",
                    "Damage States"],
            axis=1,
        ).values

        num_components = len(data)
        means_fr  = data[:, :n_ds]
        covs_fr   = data[:, n_ds:2 * n_ds]
        means_cost = data[:, 2 * n_ds:3 * n_ds] * self.conversion
        covs_cost  = data[:, 3 * n_ds:4 * n_ds]

        fragilities = {"EDP": self.edp_range, "IDs": {}}

        for item in range(num_components):
            fragilities["IDs"][item + 1] = {}
            for ds in range(n_ds):
                mean_val = means_fr[item, ds]
                cov_val  = covs_fr[item, ds]

                if mean_val == 0:
                    fragilities["IDs"][item + 1][f"DS{ds + 1}"] = np.zeros(
                        len(self.edp_range)
                    )
                else:
                    log_std  = np.sqrt(np.log(cov_val ** 2 + 1))
                    log_mean = np.exp(np.log(mean_val) - 0.5 * log_std ** 2)
                    curve = stats.norm.cdf(
                        np.log(self.edp_range / log_mean) / log_std
                    )
                    fragilities["IDs"][item + 1][f"DS{ds + 1}"] = (
                        np.nan_to_num(curve)
                    )

        return fragilities, means_cost, covs_cost

    def do_monte_carlo_simulations(self, fragilities: dict) -> dict:
        """Sample damage states via Monte Carlo for each EDP level.

        Parameters
        ----------
        fragilities : dict
            Fragility functions as returned by :meth:`fragility_function`.

        Returns
        -------
        dict
            ``{item_id: {realization_index: damage_state_array}}``.
        """
        n_ds = len(fragilities["IDs"][1])
        ds_range = np.arange(0, n_ds + 1)

        # Pre-generate all random numbers at once
        random_arrays = np.random.rand(self.realizations, len(self.edp_range))

        damage_state = {}
        for item, frag in fragilities["IDs"].items():
            damage_state[item] = {}
            for n in range(self.realizations):
                rnd = random_arrays[n]
                damage = np.zeros(len(self.edp_range), dtype=int)

                for ds in range(n_ds, 0, -1):
                    y1 = frag[f"DS{ds}"]
                    if ds == n_ds:
                        damage = np.where(rnd <= y1, ds_range[ds], damage)
                    else:
                        y2 = frag[f"DS{ds + 1}"]
                        damage = np.where(
                            (rnd >= y2) & (rnd < y1), ds_range[ds], damage
                        )

                damage_state[item][n] = damage

        return damage_state

    def validate_ds_dependence(self, damage_state: dict) -> dict:
        """Enforce correlated damage states for dependent components.

        If no ``correlation_tree`` was provided the damage states are returned
        unchanged.

        Parameters
        ----------
        damage_state : dict
            Sampled damage states as returned by
            :meth:`do_monte_carlo_simulations`.

        Returns
        -------
        dict
            Updated damage states with dependency constraints applied.
        """
        if self.correlation_tree is None:
            return damage_state

        for i in range(self.matrix.shape[0]):
            if i + 1 == self.matrix[i][0]:
                continue  # Independent component — skip

            m = int(self.matrix[i][0])   # causation component ID
            j = i + 1                    # dependent component ID
            for n in range(self.realizations):
                causation_ds  = damage_state[m][n]
                correlated_ds = damage_state[j][n]

                temp = np.zeros(causation_ds.shape)
                for ds in range(1, self.matrix.shape[1]):
                    temp[causation_ds == ds - 1] = self.matrix[j - 1][ds]

                damage_state[j][n] = np.maximum(correlated_ds, temp)

        return damage_state

    def calculate_costs(self,
                        damage_state: dict,
                        means_cost: np.ndarray,
                        covs_cost: np.ndarray) -> tuple:
        """Evaluate repair costs for each component at every EDP level.

        Parameters
        ----------
        damage_state : dict
            Sampled damage states from :meth:`do_monte_carlo_simulations`.
        means_cost : np.ndarray
            Shape ``(n_components, n_ds)`` — mean cost per DS.
        covs_cost : np.ndarray
            Shape ``(n_components, n_ds)`` — cost CoV per DS.

        Returns
        -------
        total_loss_storey : dict
            ``{realization: loss_array}`` — absolute storey loss.
        total_loss_storey_ratio : dict
            ``{realization: loss_ratio_array}`` — storey loss normalised by
            replacement cost.
        repair_cost : dict
            ``{item_id: {realization: cost_array}}`` — per-component costs.

        Raises
        ------
        ValueError
            If ``replacement_cost`` is zero or ``None``.
        """
        num_ds = means_cost.shape[1]
        quantities = self.component_data["Quantity"]

        repair_cost = {}
        for item in damage_state:
            idx = int(item) - 1
            repair_cost[item] = {}
            for n in range(self.realizations):
                for ds in range(num_ds + 1):
                    if ds == 0:
                        repair_cost[item][n] = np.where(
                            damage_state[item][n] == 0, 0, -1
                        )
                    else:
                        best_fit = (
                            self.component_data.iloc[item - 1][
                                f"DS{ds}, Best Fit"
                            ].lower()
                        )
                        mu  = means_cost[idx][ds - 1]
                        cov = covs_cost[idx][ds - 1]
                        idx_list = np.where(damage_state[item][n] == ds)[0]

                        for idx_repair in idx_list:
                            if best_fit == "lognormal":
                                # Lognormal is always positive — sample directly,
                                # no rejection loop needed.
                                std_log = np.sqrt(np.log(1.0 + cov ** 2))
                                m_log   = np.log(mu) - 0.5 * std_log ** 2
                                a = np.random.lognormal(m_log, std_log)
                            else:
                                # Normal can yield negatives — resample until positive.
                                a = np.random.normal(mu, cov * mu)
                                while a < 0:
                                    a = np.random.normal(mu, cov * mu)

                            repair_cost[item][n][idx_repair] = a

        # Aggregate to storey-level totals
        total_repair_cost = {
            item: {
                n: repair_cost[item][n] * quantities.iloc[item - 1]
                for n in range(self.realizations)
            }
            for item in damage_state
        }

        total_loss_storey = {}
        for n in range(self.realizations):
            total_loss_storey[n] = sum(
                total_repair_cost[item][n] for item in damage_state
            )

        if not self.replacement_cost:
            raise ValueError(
                "replacement_cost must be a non-zero positive value."
            )

        total_loss_storey_ratio = {
            n: total_loss_storey[n] / self.replacement_cost
            for n in range(self.realizations)
        }

        return total_loss_storey, total_loss_storey_ratio, repair_cost

    def perform_regression(self,
                           loss: dict,
                           loss_ratio: dict,
                           regression_type: str = None,
                           percentiles: List[float] = None) -> tuple:
        """Fit a regression model to the simulated loss data.

        If ``regression_type`` is ``None`` all supported models are tried and
        the one with the lowest combined error is selected.

        Parameters
        ----------
        loss : dict
            Raw storey-loss arrays ``{realization: array}``.
        loss_ratio : dict
            Normalised storey-loss ratio arrays ``{realization: array}``.
        regression_type : str, optional
            One of ``'weibull'``, ``'papadopoulos'``, ``'gpd'``,
            ``'lognormal'``.  If ``None``, all models are tried.
        percentiles : list of float, optional
            Quantiles to compute. Default ``[0.16, 0.50, 0.84]``.

        Returns
        -------
        losses : dict
            Quantile tables for ``'loss'`` and ``'loss_ratio'``.
        losses_fitted : dict
            Fitted loss-ratio arrays for each quantile and the mean.
        fitting_parameters : dict
            Fitted parameters per quantile.
        error_max : float
            Maximum regression error (%).
        error_cum : float
            Cumulative regression error (%).

        Raises
        ------
        ValueError
            If ``regression_type`` is not a supported model name.
        """
        percentiles = percentiles or [0.16, 0.50, 0.84]

        loss_df       = pd.DataFrame.from_dict(loss)
        loss_ratio_df = pd.DataFrame.from_dict(loss_ratio)

        losses = {
            "loss":       loss_df.quantile(percentiles, axis=1),
            "loss_ratio": loss_ratio_df.quantile(percentiles, axis=1),
        }
        losses["loss"].loc["mean"]       = loss_df.mean(axis=1)
        losses["loss_ratio"].loc["mean"] = loss_ratio_df.mean(axis=1)

        edp_range = (
            self.edp_range * 100
            if self.edp in ("idr", "psd")
            else self.edp_range
        )

        fitting_functions = {
            "weibull": {
                "func": lambda x, a, b, c: a * (1 - np.exp(-((x / b) ** c))),
                "p0": [1.0, 1.0, 1.0],
            },
            "papadopoulos": {
                "func": (
                    lambda x, a, b, c, d, e:
                    e * (x ** a) / (b ** a + x ** a)
                    + (1 - e) * (x ** c) / (d ** c + x ** c)
                ),
                "p0": [1.0, 1.0, 1.0, 1.0, 0.5],
            },
            "gpd": {
                "func": lambda x, c, loc, scale: genpareto.cdf(
                    x, c, loc=loc, scale=scale
                ),
                "p0": [0.1, 0.0, 1.0],
            },
            "lognormal": {
                "func": lambda x, mean, sigma: lognorm.cdf(
                    x, sigma, scale=np.exp(mean)
                ),
                "p0": [1.0, 1.0],
            },
        }

        if regression_type is None:
            best = {
                "losses_fitted": None,
                "fitting_parameters": None,
                "error_max": float("inf"),
                "error_cum": float("inf"),
            }
            for reg_type, fn in fitting_functions.items():
                try:
                    lf, fp, em, ec = self._fit_regression(
                        losses, edp_range, fn, percentiles
                    )
                    if em < best["error_max"] and ec < best["error_cum"]:
                        best.update(
                            losses_fitted=lf,
                            fitting_parameters=fp,
                            error_max=em,
                            error_cum=ec,
                        )
                        self.regression = reg_type
                except Exception as exc:
                    print(f"Regression failed for '{reg_type}': {exc}")

            return (
                losses,
                best["losses_fitted"],
                best["fitting_parameters"],
                best["error_max"],
                best["error_cum"],
            )

        if regression_type not in fitting_functions:
            raise ValueError(
                f"Regression type '{regression_type}' is not supported. "
                f"Choose from: {list(fitting_functions)}."
            )

        lf, fp, em, ec = self._fit_regression(
            losses, edp_range, fitting_functions[regression_type], percentiles
        )
        return losses, lf, fp, em, ec

    def estimate_accuracy(self,
                          y: np.ndarray,
                          yhat: np.ndarray) -> tuple:
        """Compute max and cumulative regression errors as percentages.

        Parameters
        ----------
        y : np.ndarray
            Observed (true) values.
        yhat : np.ndarray
            Predicted values.

        Returns
        -------
        error_max : float
            Largest absolute error as a percentage of ``max(y)``.
        error_cum : float
            Sum of absolute errors weighted by ``edp_bin``, as a percentage
            of ``max(y)``.
        """
        y    = np.asarray(y)
        yhat = np.asarray(yhat)
        abs_error = np.abs(y - yhat)
        max_y = np.max(y)
        error_max = np.max(abs_error) / max_y * 100
        error_cum = self.edp_bin * np.sum(abs_error) / max_y * 100
        return error_max, error_cum

    def transform_output(self,
                         losses_fitted: dict,
                         typology: str = None) -> dict:
        """Convert fitted SLF results into the standard output dictionary.

        Parameters
        ----------
        losses_fitted : dict
            Fitted loss functions as returned by :meth:`perform_regression`.
        typology : str, optional
            Component type label (e.g. ``'PSD, NS'``). Default ``None``.

        Returns
        -------
        dict
            SLF record with keys ``'Directionality'``, ``'Component-type'``,
            ``'Storey'``, ``'edp'``, ``'edp_range'``, and ``'slf'``.
        """
        return {
            "Directionality": self.directionality,
            "Component-type": typology,
            "Storey":         self.storey,
            "edp":            self.edp,
            "edp_range":      list(self.edp_range),
            "slf":            list(losses_fitted["mean"]),
        }

    def generate(self) -> tuple:
        """Generate Storey Loss Functions for all performance groups.

        Orchestrates the full SLF pipeline:

        1. Compute component fragility and consequence functions.
        2. Sample damage states via Monte Carlo simulation.
        3. Enforce correlated damage state constraints.
        4. Calculate repair costs per group.
        5. Fit regression models to the simulated loss data.
        6. Transform results into the output format.
        7. Compute empirical statistics (16th, median, 84th percentile).

        Returns
        -------
        out : dict
            ``{group_label: slf_dict}`` — one SLF record per performance group.
        cache : dict
            Intermediate data for each group (fragilities, losses, fitted
            parameters, empirical statistics, etc.).
        """
        out, cache = {}, {}

        fragilities, means_cost, covs_cost = self.fragility_function()
        damage_state = self.do_monte_carlo_simulations(fragilities)
        damage_state = self.validate_ds_dependence(damage_state)

        for group, component_data_group in self.component_groups.items():
            if component_data_group.empty:
                continue

            # Resolve typology label
            if isinstance(self.typology, dict):
                typology = self.typology[group].lower()
            elif isinstance(self.typology, list) and self.typology:
                typology = self.typology[0].lower()
            else:
                typology = None

            # Extract group-level subsets
            item_ids = list(component_data_group["Component ID"])
            ds_group = {key: damage_state[key] for key in item_ids}
            fragilities_group = {
                "IDs": {key: fragilities["IDs"][key] for key in item_ids},
                "EDP": fragilities["EDP"],
            }

            # Run the SLF pipeline for this group
            total, ratio, repair_cost = self.calculate_costs(
                ds_group, means_cost, covs_cost
            )
            losses, losses_fitted, fitting_parameters, error_max, error_cum = (
                self.perform_regression(total, ratio, self.regression)
            )

            group_str = str(group)
            out[group_str] = self.transform_output(losses_fitted, typology)
            out[group_str]["error_max"] = error_max
            out[group_str]["error_cum"] = error_cum

            # Compute empirical statistics over all realizations
            loss_matrix = np.array(
                [total[i] for i in range(len(total))]
            )
            cache[group_str] = {
                "component":             component_data_group,
                "fragilities":           fragilities_group,
                "total_loss_storey":     total,
                "total_loss_storey_ratio": ratio,
                "repair_cost":           repair_cost,
                "damage_states":         damage_state,
                "losses":                losses,
                "slfs":                  losses_fitted,
                "fit_pars":              fitting_parameters,
                "accuracy":              [error_max, error_cum],
                "regression":            self.regression,
                "edp":                   self.edp,
                "empirical_median":      np.median(loss_matrix, axis=0),
                "empirical_16th":        np.percentile(loss_matrix, 16, axis=0),
                "empirical_84th":        np.percentile(loss_matrix, 84, axis=0),
            }

        self.cache = cache
        return out, cache
