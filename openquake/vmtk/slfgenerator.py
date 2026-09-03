import math
import warnings
import numpy as np
import pandas as pd
from scipy import stats
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
    slf_16th : List[float]
        Empirical 16th-percentile Storey Loss Function values (loss ratio).
    slf : List[float]
        Empirical median (50th-percentile) Storey Loss Function values
        (loss ratio) -- the primary SLF curve.
    slf_84th : List[float]
        Empirical 84th-percentile Storey Loss Function values (loss ratio).
    """

    directionality: Optional[int]  = Field(alias="Directionality")
    component_type: str             = Field(alias="Component-type")
    storey: Optional[Union[int, List[int]]] = Field(alias="Storey")
    edp: str
    edp_range: List[float]
    slf_16th: List[float]
    slf: List[float]
    slf_84th: List[float]


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

        Builds ``self.id_to_pos``, the sole mapping from a component's real
        ``Component ID`` to its row position in ``self.component_data``.
        Every other method identifies components by their actual ID (not
        row position) and consults this map only where it needs to index
        into a positional array or ``.iloc``.

        Raises
        ------
        ValueError
            If, after auto-assigning missing IDs, ``Component ID`` contains
            duplicate values.
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

        ids = self.component_data["Component ID"].astype(int).tolist()
        if len(set(ids)) != len(ids):
            dupes = sorted({cid for cid in ids if ids.count(cid) > 1})
            raise ValueError(
                f"Duplicate 'Component ID' value(s) after auto-assigning "
                f"missing IDs: {dupes}. Provide explicit, unique Component "
                "ID values for every row, or leave them all blank to have "
                "1..n auto-assigned."
            )
        self.id_to_pos = {cid: pos for pos, cid in enumerate(ids)}

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

        The tree is joined to ``component_data`` by ``Component ID`` (not
        row order), via ``correlation_tree.set_index('ID').loc[component_ids]``
        -- so the two DataFrames may list components in different orders.

        The matrix has shape
        ``(n_components, n_damage_states + 1)`` where the first column stores
        the causation component ID and the remaining columns store the minimum
        damage state required on the causation component before the dependent
        component sustains damage. Row ``i`` of the matrix corresponds to
        the component whose ID is ``self.correlation_item_ids[i]``, which is
        also persisted for use by :meth:`validate_ds_dependence`.

        Notes
        -----
        Both the component inventory and the correlation tree are matched by
        ``Component ID``/``ID`` -- every ``Component ID`` in the inventory
        must have a matching row in the tree (enforced by
        :meth:`_validate_correlation_tree_schema`, which also checks that no
        minimum DS assigned in the tree exceeds the DS range actually
        defined for that component).
        """
        damage_states = list(self.component_data["Damage States"])
        component_ids = self.component_data["Component ID"].astype(int).tolist()

        self._validate_correlation_tree_schema(damage_states, component_ids)

        tree_by_id = self.correlation_tree.set_index("ID")
        correlation_data = tree_by_id.loc[component_ids].reset_index().values
        # Row i now corresponds to component_data row i by Component ID,
        # regardless of the correlation tree's own row order.

        item_ids = correlation_data[:, 0]
        self.correlation_item_ids = item_ids.astype(int)
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

    def _validate_correlation_tree_schema(self, damage_states, component_ids):
        """Validate the correlation tree against the component inventory.

        Parameters
        ----------
        damage_states : list of int
            Number of damage states for each component, in the same row
            order as ``component_ids``.
        component_ids : list of int
            ``component_data['Component ID']`` values, in row order.

        Raises
        ------
        ValueError
            On duplicate IDs, insufficient columns, DS range violations, or
            a component inventory ID missing from the correlation tree.
        """
        corr_dict = self.correlation_tree.to_dict(orient="records")

        id_set = set()
        for row in corr_dict:
            model = correlation_tree_model.model_validate(row)
            if model.ID in id_set:
                raise ValueError(f"Duplicate ITEM: {model.ID}")
            id_set.add(model.ID)

        missing = [cid for cid in component_ids if cid not in id_set]
        if missing:
            raise ValueError(
                "The correlation tree is missing a row for Component "
                f"ID(s) {missing} present in the component inventory. "
                "Every component needs a correlation-tree row (mark it "
                "'Independent' if it has no forced dependency)."
            )

        if len(self.correlation_tree.keys()) < max(damage_states) + 3:
            raise ValueError(
                "Unexpected (fewer) number of features in the correlations "
                "DataFrame."
            )

        tree_by_id = self.correlation_tree.set_index("ID")
        for cid, n_ds in zip(component_ids, damage_states):
            row = tree_by_id.loc[cid]
            for feature in tree_by_id.columns:
                if str(row[feature]) == f"DS{n_ds + 1}":
                    raise ValueError(
                        "MIN DS in the correlation tree must not exceed the "
                        "possible DS defined for the element."
                    )

    # -----------------------------------------------------------------------
    # Public methods
    # -----------------------------------------------------------------------

    def fragility_function(self) -> tuple:
        """Derive lognormal fragility functions for all components.

        Returns
        -------
        fragilities : dict
            Keys ``'EDP'`` (np.ndarray) and ``'IDs'`` (nested dict mapping
            Component ID → DS label → exceedance probability array).
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
        component_ids = self.component_data["Component ID"].astype(int).to_numpy()

        num_components = len(data)
        means_fr  = data[:, :n_ds]
        covs_fr   = data[:, n_ds:2 * n_ds]
        means_cost = data[:, 2 * n_ds:3 * n_ds] * self.conversion
        covs_cost  = data[:, 3 * n_ds:4 * n_ds]

        fragilities = {"EDP": self.edp_range, "IDs": {}}

        for pos in range(num_components):
            cid = int(component_ids[pos])
            fragilities["IDs"][cid] = {}
            for ds in range(n_ds):
                mean_val = means_fr[pos, ds]
                cov_val  = covs_fr[pos, ds]

                if mean_val == 0:
                    fragilities["IDs"][cid][f"DS{ds + 1}"] = np.zeros(
                        len(self.edp_range)
                    )
                else:
                    log_std  = np.sqrt(np.log(cov_val ** 2 + 1))
                    log_mean = np.exp(np.log(mean_val) - 0.5 * log_std ** 2)
                    curve = stats.norm.cdf(
                        np.log(self.edp_range / log_mean) / log_std
                    )
                    fragilities["IDs"][cid][f"DS{ds + 1}"] = (
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
        n_ds = len(next(iter(fragilities["IDs"].values())))
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
            own_id = int(self.correlation_item_ids[i])
            if own_id == int(self.matrix[i][0]):
                continue  # Independent component — skip

            m = int(self.matrix[i][0])   # causation component ID
            j = own_id                    # dependent component ID (from the tree)
            for n in range(self.realizations):
                causation_ds  = damage_state[m][n]
                correlated_ds = damage_state[j][n]

                temp = np.zeros(causation_ds.shape)
                for ds in range(1, self.matrix.shape[1]):
                    temp[causation_ds == ds - 1] = self.matrix[i][ds]

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
            pos = self.id_to_pos[item]
            repair_cost[item] = {}
            for n in range(self.realizations):
                for ds in range(num_ds + 1):
                    if ds == 0:
                        repair_cost[item][n] = np.where(
                            damage_state[item][n] == 0, 0, -1
                        )
                    else:
                        best_fit = (
                            self.component_data.iloc[pos][
                                f"DS{ds}, Best Fit"
                            ].lower()
                        )
                        mu  = means_cost[pos][ds - 1]
                        cov = covs_cost[pos][ds - 1]
                        idx_list = np.where(damage_state[item][n] == ds)[0]
                        n_repair = len(idx_list)

                        if n_repair:
                            if best_fit == "lognormal":
                                # Lognormal is always positive — sample directly,
                                # no rejection loop needed.
                                std_log = np.sqrt(np.log(1.0 + cov ** 2))
                                m_log   = np.log(mu) - 0.5 * std_log ** 2
                                a_vals = np.random.lognormal(
                                    m_log, std_log, size=n_repair
                                )
                            elif cov > 0:
                                # Normal truncated at 0 -- same distribution as
                                # "resample until positive", but no unbounded
                                # rejection loop (which could hang if cov == 0
                                # and mu <= 0, since every draw would then be
                                # the same non-positive value). Drawn in one
                                # batched call per (component, realization, DS)
                                # rather than one at a time -- scipy's rv_continuous
                                # machinery has substantial per-call overhead.
                                a_std = cov * mu
                                a_vals = stats.truncnorm.rvs(
                                    (0 - mu) / a_std, np.inf,
                                    loc=mu, scale=a_std, size=n_repair,
                                )
                            else:
                                a_vals = np.full(n_repair, mu)

                            repair_cost[item][n][idx_list] = a_vals

        # Aggregate to storey-level totals
        total_repair_cost = {
            item: {
                n: repair_cost[item][n] * quantities.iloc[self.id_to_pos[item]]
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

    def transform_output(self,
                         slf_16th: np.ndarray,
                         slf_median: np.ndarray,
                         slf_84th: np.ndarray,
                         typology: str = None) -> dict:
        """Build the SLF output record for a performance group.

        Parameters
        ----------
        slf_16th, slf_median, slf_84th : np.ndarray
            Empirical 16th/50th/84th percentile of the storey loss RATIO,
            evaluated at every point in ``self.edp_range``.
        typology : str, optional
            Component type label (e.g. ``'PSD, NS'``). Default ``None``.

        Returns
        -------
        dict
            SLF record with keys ``'Directionality'``, ``'Component-type'``,
            ``'Storey'``, ``'edp'``, ``'edp_range'``, ``'slf_16th'``,
            ``'slf'`` (the empirical median -- the primary SLF curve), and
            ``'slf_84th'``.
        """
        return {
            "Directionality": self.directionality,
            "Component-type": typology,
            "Storey":         self.storey,
            "edp":            self.edp,
            "edp_range":      list(self.edp_range),
            "slf_16th":       list(slf_16th),
            "slf":            list(slf_median),
            "slf_84th":       list(slf_84th),
        }

    def generate(self) -> tuple:
        """Generate Storey Loss Functions for all performance groups.

        Orchestrates the full SLF pipeline:

        1. Compute component fragility and consequence functions.
        2. Sample damage states via Monte Carlo simulation.
        3. Enforce correlated damage state constraints.
        4. Calculate repair costs per group.
        5. Compute the empirical 16th/50th/84th percentile of the storey
           loss ratio across all Monte Carlo realizations -- this IS the
           Storey Loss Function. No parametric curve is fitted to it.

        Returns
        -------
        out : dict
            ``{group_label: slf_dict}`` -- one SLF record per performance
            group, with ``'slf_16th'``, ``'slf'`` (median), and
            ``'slf_84th'`` giving the empirical percentiles of the storey
            loss ratio vs. ``'edp_range'``.
        cache : dict
            Intermediate data for each group: fragilities, raw
            per-realization losses (both absolute ``'total_loss_storey'``
            and ratio ``'total_loss_storey_ratio'`` units), repair costs,
            the group-sliced damage states, and the same empirical
            percentiles as ``out`` (mirrored here as
            ``'empirical_16th'``/``'empirical_median'``/``'empirical_84th'``).
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

            # Empirical percentiles of the storey loss RATIO -- the
            # canonical (dimensionless) Storey Loss Function. Computed
            # once and reused for both 'out' and 'cache' so they never
            # diverge.
            ratio_matrix = np.array([ratio[i] for i in range(len(ratio))])
            slf_16th   = np.percentile(ratio_matrix, 16, axis=0)
            slf_median = np.median(ratio_matrix, axis=0)
            slf_84th   = np.percentile(ratio_matrix, 84, axis=0)

            group_str = str(group)
            out[group_str] = self.transform_output(
                slf_16th, slf_median, slf_84th, typology
            )

            cache[group_str] = {
                "component":               component_data_group,
                "fragilities":             fragilities_group,
                "total_loss_storey":       total,
                "total_loss_storey_ratio": ratio,
                "repair_cost":             repair_cost,
                "damage_states":           ds_group,
                "edp":                     self.edp,
                "empirical_16th":          slf_16th,
                "empirical_median":        slf_median,
                "empirical_84th":          slf_84th,
            }

        self.cache = cache
        return out, cache
