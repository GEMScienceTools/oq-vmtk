"""
Storey Loss Function (SLF) generator.

A Storey Loss Function gives, for the components of one storey that are
sensitive to one engineering demand parameter (EDP: peak storey drift or peak
floor acceleration), the repair cost as a fraction of their replacement cost,
as a function of that demand. ``slfgenerator`` builds it from a component
inventory in the FEMA P-58 format: quantities, fragility functions, damage
state hierarchies and repair-cost distributions.

For every EDP value it returns the mean, standard deviation and
16th/50th/84th percentiles of the storey loss ratio, from Monte Carlo
realisations in which the loss ratio stays within [0, 1].

Inventory (one row per component)
---------------------------------
Required columns:

``Component ID``
    Unique positive integer (blank IDs are numbered 1..n).
``EDP``, ``Typology``
    Demand parameter (``PSD``/``IDR`` or ``PFA``) and component type (``S``
    structural, ``NS`` nonstructural); together they form the default
    performance groups.
``Quantity``
    Number of costing units on the storey.
``Damage States``
    Number of damage states ``n`` of the component.
``DSk, Median``, ``DSk, Total Dispersion``
    Median (in EDP units) and logarithmic standard deviation of the lognormal
    fragility of the limit state that damage state ``k`` belongs to.
``DSk, Cost``, ``DSk, Cost Dispersion``, ``DSk, Best Fit``
    Repair cost per unit in damage state ``k``. ``Lognormal``: median and
    logarithmic standard deviation. ``Normal`` (the default when blank): mean
    and coefficient of variation; the distribution is truncated at zero.
    These are the FEMA P-58 conventions.

Optional columns:

``Description``, ``Performance Group``
    As before; explicit performance groups override the ``(EDP, Typology)``
    grouping when more than one is given.
``DS Hierarchy``
    FEMA P-58 damage state hierarchy, e.g. ``Seq(DS1,DS2)``,
    ``Seq(DS1,MutEx(DS2,DS3))`` or ``Simul(DS1,DS2,DS3,DS4)``. Each item of
    the sequence is one limit state, with one fragility; a ``MutEx`` group
    is a limit state whose damage states are mutually exclusive, a ``Simul``
    group one whose damage states can occur together. Default: all damage
    states sequential, ``Seq(DS1,...,DSn)``.
``DSk, Probability``
    Probability of damage state ``k`` within its limit state: for ``MutEx``
    the probability that it is the one that occurs (they sum to one); for
    ``Simul`` the probability that it occurs, independently of the others,
    given at least one does. Ignored for a limit state with one damage state.
``Correlated``
    FEMA P-58 "Correlated?" flag. ``YES`` (default): all units of the
    component on the storey are in the same damage state. ``NO``: every unit
    is damaged independently; the quantity is split into ``ceil(Quantity)``
    equal blocks, each with its own capacity and damage state. The default
    suits quantities expressed per unit floor area; ``NO`` requires
    ``Quantity`` to count the units on the storey.
``Replacement Cost``
    Expected cost of replacing one unit. Default: the largest expected
    repair cost of any damage outcome of the component (for a ``Simul``
    limit state, all of its damage states together).

Simulation
----------
In each realisation:

- every unit (or block of units) draws one capacity, held over the whole EDP
  range: its limit state at an EDP is the number of fragility curves of the
  component exceeded at that EDP;
- the damage state reached within a ``MutEx`` or ``Simul`` limit state is
  drawn once per unit and limit state;
- the unit repair costs of a component are drawn at one probability level
  per realisation, shared by its damage states and its units: one price
  level for repairing and replacing the same item (FEMA P-58 samples one
  unit cost per damage state and performance group). The cost of the
  replacement outcome (the damage outcome with the largest expected cost)
  at that level is the unit replacement cost of the realisation; no unit is
  repaired for more than it;
- a correlation tree forces a dependent component to at least a given limit
  state when its causation component reaches a given limit state. With
  per-unit sampling, dependent unit ``b`` of ``n_d`` is hosted by causation
  unit ``floor(b * n_c / n_d)`` of ``n_c``, so each causation unit carries an
  equal share of the dependent units.

Each component draws from its own random stream, in inventory order, so
models that differ only by their correlation tree share their random numbers
and the effect of the tree shows realisation by realisation.

The storey loss ratio of a realisation is its repair cost divided by the
replacement cost of the same components at the same unit costs. It lies in
[0, 1] and is 1 when every unit is in its replacement outcome; the price
level of a component, which scales repair and replacement alike, does not
change it. Its mean is the SLF; the exact ratio of the expected repair cost
to the expected replacement cost is also returned, as a check (they differ
by about 0.01 at most, the two being different statistics).

Not modelled: economies of scale (the unit cost does not depend on the
number of damaged units).

References
----------
1. Ramirez and Miranda (2009). Building-Specific Loss Estimation Methods &
   Tools for Simplified PBEE. John A. Blume EEC, Stanford University.
2. FEMA (2018). Seismic Performance Assessment of Buildings, FEMA P-58,
   2nd edition, Volume 1: Methodology. Federal Emergency Management Agency.
3. Shahnazaryan D, O'Reilly GJ, Monteiro R. (2021). Story loss functions for
   seismic design and assessment. Earthquake Spectra, 37(4): 2813-2839.
   https://doi.org/10.1177/87552930211023523
4. Shahnazaryan D, O'Reilly GJ, Monteiro R. (2021). Development of a
   Python-Based Storey Loss Function Generator. COMPDYN 2021.
   https://doi.org/10.7712/120121.8659.18567
"""
import math
import re
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator
from scipy import special


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

    @field_validator("Component_ID")
    @classmethod
    def validate_id(cls, v):
        if v is not None and v < 0:
            raise ValueError("Component ID must be a positive integer")
        return v

    @field_validator("Quantity")
    @classmethod
    def validate_quantity(cls, v):
        if not np.isfinite(v) or v < 0:
            raise ValueError("Quantity must be a non-negative number")
        return v

    @field_validator("Damage_States")
    @classmethod
    def validate_damage_states(cls, v):
        if v < 1:
            raise ValueError("Damage States must be at least 1")
        return v

    @field_validator("Performance_Group", "Component_ID", "Description",
                     mode="before")
    @classmethod
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
        ``'Independent'`` or the ``Component ID`` of the causation component.
    """

    ID: int
    dependent_on_item: str = Field(alias="DEPENDENT ON ITEM")

    @field_validator("ID")
    @classmethod
    def validate_id(cls, vid):
        if vid < 0:
            raise ValueError("Component ID must be a positive integer")
        return vid

    @field_validator("dependent_on_item", mode="before")
    @classmethod
    def as_text(cls, v):
        if isinstance(v, float) and v.is_integer():
            return str(int(v))
        return str(v)


class slf_model(BaseModel):
    """Storey Loss Function output record of one performance group.

    Attributes
    ----------
    directionality : Optional[int]
        Analysis directionality flag.
    component_type : Optional[str]
        Component type label (e.g., ``'ns'``).
    storey : Optional[Union[int, List[int]]]
        Storey level(s) covered by this SLF.
    edp : str
        Engineering Demand Parameter label.
    edp_range : List[float]
        EDP values over which the SLF is defined.
    slf : List[float]
        Mean storey loss ratio: the Storey Loss Function.
    slf_std : List[float]
        Standard deviation of the storey loss ratio.
    slf_cov : List[float]
        ``slf_std / slf`` (0 where ``slf`` is 0).
    slf_16th, slf_50th, slf_84th : List[float]
        Empirical 16th/50th/84th percentiles of the storey loss ratio.
    replacement_cost : float
        ``replacement_cost`` if given, otherwise the expected replacement
        cost of the group's components.
    """

    directionality: Optional[int] = Field(alias="Directionality")
    component_type: Optional[str] = Field(alias="Component-type")
    storey: Optional[Union[int, List[int]]] = Field(alias="Storey")
    edp: str
    edp_range: List[float]
    slf: List[float]
    slf_std: List[float]
    slf_cov: List[float]
    slf_16th: List[float]
    slf_50th: List[float]
    slf_84th: List[float]
    replacement_cost: float


# ---------------------------------------------------------------------------
# Inventory parsing helpers
# ---------------------------------------------------------------------------

_DS_FIELDS = ("Median", "Total Dispersion", "Cost", "Cost Dispersion",
              "Best Fit")
_YES = {"yes", "y", "true", "1", "1.0"}
_NO = {"no", "n", "false", "0", "0.0"}
_UNSET = {"", "nan", "none", "independent", "undamaged"}


def _blank(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return isinstance(value, str) and not value.strip()


def _split_top_level(text: str) -> List[str]:
    """Split ``text`` at the commas that are not inside parentheses."""
    items, depth, current = [], 0, ""
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            items.append(current)
            current = ""
        else:
            current += ch
    items.append(current)
    return items


def _parse_hierarchy(text, n_ds: int, cid: int) -> List[tuple]:
    """Limit states of a FEMA P-58 damage state hierarchy.

    Returns ``[(kind, [damage states]), ...]`` in sequence order, with
    ``kind`` one of ``'single'``, ``'mutex'``, ``'simul'`` and damage states
    numbered from 1.
    """
    if _blank(text):
        return [("single", [k]) for k in range(1, n_ds + 1)]
    s = re.sub(r"\s+", "", str(text)).lower()
    m = re.fullmatch(r"seq\((.*)\)", s)
    body = m.group(1) if m else s
    limit_states = []
    for item in _split_top_level(body):
        single = re.fullmatch(r"ds(\d+)", item)
        group = re.fullmatch(r"(mutex|simul)\(((?:ds\d+,)*ds\d+)\)", item)
        if single:
            limit_states.append(("single", [int(single.group(1))]))
        elif group:
            ds = [int(d[2:]) for d in group.group(2).split(",")]
            kind = group.group(1) if len(ds) > 1 else "single"
            limit_states.append((kind, ds))
        else:
            raise ValueError(
                f"Component {cid}: cannot read DS Hierarchy {text!r} "
                "(expected e.g. 'Seq(DS1,MutEx(DS2,DS3))')."
            )
    listed = [d for _, ds in limit_states for d in ds]
    if listed != list(range(1, n_ds + 1)):
        raise ValueError(
            f"Component {cid}: DS Hierarchy {text!r} must list DS1..DS{n_ds} "
            "once each, in order."
        )
    return limit_states


def _flag(value, cid: int) -> bool:
    """``Correlated`` column value as a boolean (blank means correlated)."""
    if _blank(value):
        return True
    text = str(value).strip().lower()
    if text in _YES:
        return True
    if text in _NO:
        return False
    raise ValueError(
        f"Component {cid}: Correlated must be YES or NO, got {value!r}."
    )


def _state_index(value, cid: int, column: str) -> int:
    """Limit state named in a correlation tree cell (0 when unset)."""
    if _blank(value) or str(value).strip().lower() in _UNSET:
        return 0
    m = re.fullmatch(r"(?:ds|ls)(\d+)", str(value).strip().lower())
    if not m:
        raise ValueError(
            f"Correlation tree row {cid}, column {column}: expected "
            f"'Independent' or 'DSk', got {value!r}."
        )
    return int(m.group(1))


def _mean_unit_cost(theta0, theta1, family):
    """Mean of a unit repair cost: lognormal (median, beta) or normal
    (mean, CoV) truncated at zero."""
    if theta0 <= 0:
        return 0.0
    if family == "lognormal":
        return theta0 * math.exp(0.5 * theta1 ** 2)
    if theta1 <= 0:
        return theta0
    a = -1.0 / theta1
    hazard = math.exp(-0.5 * a * a) / math.sqrt(2 * math.pi) / special.ndtr(-a)
    return theta0 * (1.0 + theta1 * hazard)


def _sample_unit_cost(theta0, theta1, family, u):
    """Unit repair costs at probability levels ``u`` (inverse transform)."""
    if theta0 <= 0:
        return np.zeros_like(u)
    if family == "lognormal":
        return theta0 * np.exp(theta1 * special.ndtri(u))
    if theta1 <= 0:
        return np.full_like(u, theta0)
    p0 = special.ndtr(-1.0 / theta1)
    return theta0 * (1.0 + theta1 * special.ndtri(p0 + u * (1.0 - p0)))


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class slfgenerator:
    """Storey Loss Function (SLF) generator for storey-based loss assessment.

    Builds Storey Loss Functions from a FEMA P-58 style component inventory
    by Monte Carlo simulation, with independent component (and, where not
    correlated, unit) capacities, explicit mutually exclusive and
    simultaneous damage states, optional correlation trees, and a storey
    loss ratio in [0, 1] in every realisation. See the module documentation
    for the inventory format and the simulation.

    References
    ----------
    1. Ramirez and Miranda (2009). *Building-Specific Loss Estimation Methods
       & Tools for Simplified PBEE*. John A. Blume EEC, Stanford University.
    2. FEMA (2018). *Seismic Performance Assessment of Buildings*, FEMA P-58,
       2nd edition.
    3. Shahnazaryan D, O'Reilly GJ, Monteiro R. (2021). Story loss functions
       for seismic design and assessment. *Earthquake Spectra*, 37(4):
       2813–2839. https://doi.org/10.1177/87552930211023523
    4. Shahnazaryan D, O'Reilly GJ, Monteiro R. (2021). Development of a
       Python-Based Storey Loss Function Generator. *COMPDYN 2021*.
       https://doi.org/10.7712/120121.8659.18567

    Acknowledgements
    ----------------
    Based on the original work by Dr. Davit Shahnazaryan:
    https://github.com/davitshahnazaryan3/SLFGenerator
    """

    def __init__(self,
                 component_data: pd.DataFrame,
                 edp: str,
                 correlation_tree: pd.DataFrame = None,
                 typology: Union[List[str], Dict[str, str]] = None,
                 edp_range: Union[List[float], np.ndarray] = None,
                 edp_bin: float = None,
                 grouping_flag: bool = True,
                 conversion: float = 1.0,
                 realizations: int = 1000,
                 replacement_cost: float = None,
                 storey: Union[int, List[int]] = None,
                 directionality: int = None,
                 seed: Union[int, np.random.Generator] = None):
        """Initialise the SLF generator.

        Parameters
        ----------
        component_data : pandas.DataFrame
            Component inventory (see the module documentation for the
            columns). Not modified.
        edp : str
            Engineering Demand Parameter; ``'PSD'`` or ``'IDR'`` (peak storey
            drift) or ``'PFA'`` (peak floor acceleration).
        correlation_tree : pandas.DataFrame, optional
            Correlation tree: columns ``ID``, ``DEPENDENT ON ITEM`` and
            ``DS0``..``DSn``. For a dependent component, column ``DSk`` gives
            the limit state (``'DSj'``) the component is forced to at least
            when its causation component is in limit state ``k``
            (``'Independent'``: no forcing). For components whose damage
            states are all sequential (the default) limit state ``k`` is
            damage state ``k``. Default ``None`` (independent components).
        typology : list of str or dict, optional
            Component-type label written to the output records. Default
            ``None``.
        edp_range : array-like, optional
            EDP values at which the SLF is evaluated (non-negative; zero is
            allowed). Default: 0 to 0.5 by ``edp_bin`` for drift, 0 to 5 g by
            ``edp_bin`` for acceleration. The array passed is not modified.
        edp_bin : float, optional
            Step of the default EDP range. Default 0.001 (drift) or 0.05 g
            (acceleration).
        grouping_flag : bool, optional
            Whether to group components by ``(EDP, Typology)`` or by explicit
            ``Performance Group``. If ``False`` all components form one group.
            Default ``True``.
        conversion : float, optional
            Factor applied to all repair and replacement costs of the
            inventory (e.g. a currency conversion). Default ``1.0``.
        realizations : int, optional
            Number of Monte Carlo realisations. Default ``1000``.
        replacement_cost : float, optional
            Fixed cost the loss of every group is divided by, in converted
            units; the ratio is then not bounded by 1. Default ``None``: each
            realisation of a group is divided by the replacement cost of its
            own components at that realisation's unit costs (``Quantity``
            times the unit replacement cost, summed), so that the SLF is a
            loss ratio in [0, 1].
        storey : int or List[int], optional
            Storey level(s), copied to the output. Default ``None``.
        directionality : int, optional
            Directionality flag, copied to the output. Default ``None``.
        seed : int or numpy.random.Generator, optional
            Seed of the Monte Carlo simulation. The same seed and inputs give
            the same realisations; with the same seed, adding or changing a
            correlation tree leaves every component's random numbers
            unchanged. Default ``None`` (unseeded).

        Raises
        ------
        ValueError
            On an invalid EDP, EDP range, inventory, correlation tree or
            setting.
        """
        self.edp = edp.lower()
        self.typology = typology
        self.edp_bin = edp_bin
        self.grouping_flag = grouping_flag
        self.conversion = float(conversion)
        self.realizations = int(realizations)
        self.replacement_cost = replacement_cost
        self.storey = storey
        self.directionality = directionality
        self.seed = seed

        if self.realizations < 2:
            raise ValueError("realizations must be at least 2.")
        if not (np.isfinite(self.conversion) and self.conversion > 0):
            raise ValueError("conversion must be a positive number.")
        if replacement_cost is not None and not (
                np.isfinite(replacement_cost) and replacement_cost > 0):
            raise ValueError(
                "replacement_cost must be a positive number or None."
            )

        self.component_data = component_data.copy()
        for column in ("EDP", "Typology"):
            if column in self.component_data:
                self.component_data[column] = self.component_data[column].map(
                    lambda s: s.lower() if isinstance(s, str) else s
                )

        self.edp_range = self._define_edp_range(edp_range)
        self._get_component_data()

        self.correlation_tree = correlation_tree
        self.matrix = None
        self.correlation_item_ids = None
        self.causation = {}
        self._get_correlation_tree()

        self._group_components()

    # -----------------------------------------------------------------------
    # Private helpers: inputs
    # -----------------------------------------------------------------------

    def _define_edp_range(self, edp_range) -> np.ndarray:
        """EDP values at which the SLF is evaluated.

        Raises
        ------
        ValueError
            If ``edp`` is not ``'psd'``, ``'idr'`` or ``'pfa'``, or the range
            is empty, not one-dimensional, negative or not finite.
        """
        edp_defaults = {
            "idr": (0.1 / 100, 0.5),
            "psd": (0.1 / 100, 0.5),
            "pfa": (0.05, 5.0),
        }
        if self.edp not in edp_defaults:
            raise ValueError(
                "Incorrect EDP provided — must be 'psd', 'idr', or 'pfa'."
            )
        default_bin, range_end = edp_defaults[self.edp]
        if self.edp_bin is None:
            self.edp_bin = default_bin
        if edp_range is None:
            n_bins = int(round(range_end / self.edp_bin))
            return np.linspace(0.0, n_bins * self.edp_bin, n_bins + 1)

        values = np.array(edp_range, dtype=float)       # a copy
        if values.ndim != 1 or values.size == 0:
            raise ValueError("edp_range must be a non-empty 1-D array.")
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError("edp_range must be finite and non-negative.")
        return values

    def _ds_columns(self) -> int:
        """Number of ``DSk`` column sets, checking that each is complete."""
        columns = set(self.component_data.columns)
        ks = sorted(int(m.group(1)) for c in columns
                    if (m := re.fullmatch(r"DS(\d+), Median", c)))
        if not ks or ks != list(range(1, len(ks) + 1)):
            raise ValueError(
                "The inventory needs 'DS1, Median', 'DS2, Median', ... "
                "columns numbered from 1."
            )
        for k in ks:
            missing = [f"DS{k}, {f}" for f in _DS_FIELDS
                       if f"DS{k}, {f}" not in columns]
            if missing:
                raise ValueError(
                    "Column counts must be equal for 'Median', "
                    "'Total Dispersion', 'Cost', 'Cost Dispersion', and "
                    f"'Best Fit': missing {missing}."
                )
        return len(ks)

    def _get_component_data(self):
        """Parse and validate the component inventory.

        Missing ``Component ID`` values are filled with ``1..n`` by row
        position. Builds ``self.id_to_pos`` (``Component ID`` -> row
        position) and ``self.components`` (``Component ID`` -> parsed
        component), which hold, per component, its limit states (fragility
        median and dispersion, damage state outcomes and their
        probabilities), unit repair-cost distributions, unit replacement
        cost and blocks of units sampled independently.

        Raises
        ------
        ValueError
            On duplicate IDs, missing columns or invalid values.
        """
        df = self.component_data
        n_cols = self._ds_columns()

        id_set = set()
        for row in df.to_dict(orient="records"):
            model = component_data_model.model_validate(row)
            if model.Component_ID is not None and model.Component_ID in id_set:
                raise ValueError(f"Duplicate Component ID: {model.Component_ID}")
            id_set.add(model.Component_ID)

        df["Component ID"] = df["Component ID"].fillna(
            pd.Series(np.arange(1, len(df) + 1), index=df.index)
        )
        ids = df["Component ID"].astype(int).tolist()
        if len(set(ids)) != len(ids):
            dupes = sorted({cid for cid in ids if ids.count(cid) > 1})
            raise ValueError(
                f"Duplicate 'Component ID' value(s) after auto-assigning "
                f"missing IDs: {dupes}. Provide explicit, unique Component "
                "ID values for every row, or leave them all blank to have "
                "1..n auto-assigned."
            )
        df["Component ID"] = ids
        self.id_to_pos = {cid: pos for pos, cid in enumerate(ids)}

        self.components = {}
        for pos, (_, row) in enumerate(df.iterrows()):
            cid = ids[pos]
            n_ds = int(row["Damage States"])
            if n_ds > n_cols:
                raise ValueError(
                    f"Component {cid}: {n_ds} damage states but only "
                    f"{n_cols} 'DSk' column sets."
                )
            self.components[cid] = self._parse_component(cid, pos, row, n_ds)

    def _parse_component(self, cid, pos, row, n_ds) -> dict:
        """Limit states, costs, replacement cost and blocks of one row."""
        def value(k, field):
            return row.get(f"DS{k}, {field}")

        def number(k, field, default=None):
            v = value(k, field)
            if _blank(v):
                return default
            try:
                return float(v)
            except (TypeError, ValueError):
                raise ValueError(
                    f"Component {cid}: 'DS{k}, {field}' is not a number: {v!r}."
                ) from None

        # --- unit repair costs, per damage state
        theta0, theta1, family, mean_cost = [], [], [], []
        for k in range(1, n_ds + 1):
            t0, t1 = number(k, "Cost", 0.0), number(k, "Cost Dispersion", 0.0)
            fit = value(k, "Best Fit")
            fit = "normal" if _blank(fit) else str(fit).strip().lower()
            if fit not in ("normal", "lognormal"):
                raise ValueError(
                    f"Component {cid}: 'DS{k}, Best Fit' must be Normal or "
                    f"Lognormal, got {value(k, 'Best Fit')!r}."
                )
            if not (np.isfinite(t0) and t0 >= 0 and np.isfinite(t1) and t1 >= 0):
                raise ValueError(
                    f"Component {cid}: DS{k} cost {t0} and cost dispersion "
                    f"{t1} must be finite and non-negative."
                )
            t0 *= self.conversion
            theta0.append(t0)
            theta1.append(t1)
            family.append(fit)
            mean_cost.append(_mean_unit_cost(t0, t1, fit))
        mean_cost = np.array(mean_cost)

        # --- limit states: fragility and damage state outcomes
        limit_states = []
        hierarchy = row.get("DS Hierarchy")
        for kind, ds in _parse_hierarchy(hierarchy, n_ds, cid):
            first = ds[0]
            median, beta = number(first, "Median"), number(first, "Total Dispersion")
            if not (median and median > 0 and np.isfinite(median)
                    and beta and beta > 0 and np.isfinite(beta)):
                raise ValueError(
                    f"Component {cid}: DS{first} needs a positive median and "
                    f"total dispersion, got {median} and {beta}."
                )
            for d in ds[1:]:
                for field, ref in (("Median", median), ("Total Dispersion", beta)):
                    other = number(d, field)
                    if other is not None and not np.isclose(other, ref, rtol=1e-6):
                        raise ValueError(
                            f"Component {cid}: DS{d} shares the limit state of "
                            f"DS{first} but has a different {field.lower()} "
                            f"({other} vs {ref})."
                        )
            idx = [d - 1 for d in ds]
            if kind == "single":
                outcomes, p_outcome = [tuple(idx)], np.ones(1)
            else:
                p = np.array([number(d, "Probability", np.nan) for d in ds])
                if kind == "mutex":
                    if (not np.all(np.isfinite(p)) or np.any(p < 0)
                            or not np.isclose(p.sum(), 1.0, atol=1e-3)):
                        raise ValueError(
                            f"Component {cid}: the probabilities of mutually "
                            f"exclusive DS{ds} must be non-negative and sum to "
                            f"1, got {p.tolist()}."
                        )
                    outcomes, p_outcome = [(i,) for i in idx], p / p.sum()
                else:
                    if not np.all(np.isfinite(p)) or np.any(p <= 0) or np.any(p > 1):
                        raise ValueError(
                            f"Component {cid}: the probabilities of "
                            f"simultaneous DS{ds} must be in (0, 1], got "
                            f"{p.tolist()}."
                        )
                    outcomes, weights = [], []
                    for mask in range(1, 2 ** len(ds)):
                        bits = [(mask >> j) & 1 for j in range(len(ds))]
                        outcomes.append(tuple(i for i, b in zip(idx, bits) if b))
                        weights.append(np.prod([pj if b else 1.0 - pj
                                                for pj, b in zip(p, bits)]))
                    p_outcome = np.array(weights) / (1.0 - np.prod(1.0 - p))
            outcome_cost = np.array([mean_cost[list(o)].sum() for o in outcomes])
            limit_states.append({
                "kind": kind, "damage_states": ds, "median": median,
                "beta": beta, "outcomes": outcomes, "p_outcome": p_outcome,
                "cum_p": np.cumsum(p_outcome)[:-1],
                "outcome_mean_cost": outcome_cost,
                "mean_cost": float(p_outcome @ outcome_cost),
            })
        medians = [ls["median"] for ls in limit_states]
        if np.any(np.diff(medians) < 0):
            raise ValueError(
                f"Component {cid}: limit state medians must not decrease, "
                f"got {medians}."
            )

        # --- replacement cost per unit (converted). The replacement outcome is
        # the damage outcome with the largest expected cost (the most severe
        # one among equals); its realised cost sets the component's price
        # level in each realisation.
        replacement_outcome, top = (0, 0), -1.0
        for k, ls in enumerate(limit_states):
            for o, cost in enumerate(ls["outcome_mean_cost"]):
                if cost >= top:
                    replacement_outcome, top = (k, o), float(cost)
        given = row.get("Replacement Cost")
        if _blank(given):
            replacement = top
        else:
            replacement = float(given) * self.conversion
            if not (np.isfinite(replacement) and replacement >= 0):
                raise ValueError(
                    f"Component {cid}: Replacement Cost must be non-negative."
                )

        # --- blocks of units sampled independently
        quantity = float(row["Quantity"])
        correlated = _flag(row.get("Correlated"), cid)
        blocks = 1 if correlated else max(1, int(math.ceil(quantity - 1e-9)))
        return {
            "id": cid, "position": pos, "quantity": quantity,
            "damage_states": n_ds, "limit_states": limit_states,
            "n_ls": len(limit_states), "cost_theta0": theta0,
            "cost_theta1": theta1, "cost_family": family,
            "mean_cost": mean_cost, "replacement": replacement,
            "replacement_outcome": replacement_outcome,
            "replacement_outcome_mean": top,
            "correlated": correlated, "blocks": blocks,
            "block_quantity": quantity / blocks,
        }

    def _get_correlation_tree(self):
        """Read and validate the correlation tree.

        The tree is joined to the inventory by ``Component ID``, so the two
        may list components in different orders. Sets:

        - ``self.causation``: dependent ID -> (causation ID, ``lut``), where
          ``lut[k]`` is the limit state the dependent is forced to at least
          when the causation component is in limit state ``k``;
        - ``self.matrix``: one row per component in inventory order; column
          0 is the causation ID (the component's own ID if independent),
          column ``k + 1`` the forced limit state for causation limit state
          ``k``;
        - ``self.correlation_item_ids``: the IDs of the matrix rows;
        - ``self.order``: component IDs with every causation component
          before its dependents.

        Raises
        ------
        ValueError
            On duplicate or missing IDs, unknown causation components,
            cycles, a forced limit state beyond the dependent's limit
            states, a forcing from an undamaged causation component, or a
            forcing that decreases as the causation component gets more
            damaged. Columns beyond the causation component's limit states
            are ignored.
        """
        ids = list(self.components)
        self.order = ids
        if self.correlation_tree is None:
            return
        tree = self.correlation_tree.copy()
        tree.columns = [str(c).strip() for c in tree.columns]

        id_set = set()
        for row in tree.to_dict(orient="records"):
            model = correlation_tree_model.model_validate(row)
            if model.ID in id_set:
                raise ValueError(f"Duplicate ITEM: {model.ID}")
            id_set.add(model.ID)
        missing = [cid for cid in ids if cid not in id_set]
        if missing:
            raise ValueError(
                "The correlation tree is missing a row for Component "
                f"ID(s) {missing} present in the component inventory. "
                "Every component needs a correlation-tree row (mark it "
                "'Independent' if it has no forced dependency)."
            )

        ds_cols = sorted((int(m.group(1)), c) for c in tree.columns
                         if (m := re.fullmatch(r"DS(\d+)", c, flags=re.I)))
        n_cols = len(ds_cols)
        if [k for k, _ in ds_cols] != list(range(n_cols)):
            raise ValueError(
                "The correlation tree needs columns DS0, DS1, ... numbered "
                "from 0."
            )
        max_ls = max(c["n_ls"] for c in self.components.values())
        if n_cols < max_ls + 1:
            raise ValueError(
                "Unexpected (fewer) number of features in the correlations "
                f"DataFrame: DS0..DS{max_ls} are needed."
            )

        tree["ID"] = tree["ID"].astype(int)
        tree_by_id = tree.set_index("ID")
        self.matrix = np.zeros((len(ids), n_cols + 1))
        self.correlation_item_ids = np.array(ids, dtype=int)
        for i, cid in enumerate(ids):
            row = tree_by_id.loc[cid]
            dep = str(row["DEPENDENT ON ITEM"]).strip()
            if dep.lower() in _UNSET:
                self.matrix[i, 0] = cid
                for k, column in ds_cols:
                    if _state_index(row[column], cid, column):
                        raise ValueError(
                            f"Component {cid} is independent but its "
                            f"correlation tree column {column} forces a "
                            "damage state."
                        )
                continue
            try:
                cause = int(float(dep))
            except ValueError:
                raise ValueError(
                    f"Component {cid}: DEPENDENT ON ITEM must be "
                    f"'Independent' or a Component ID, got {dep!r}."
                ) from None
            if cause not in self.components or cause == cid:
                raise ValueError(
                    f"Component {cid}: causation component {cause} is not "
                    "another component of the inventory."
                )
            n_cause = self.components[cause]["n_ls"]
            n_own = self.components[cid]["n_ls"]
            lut = np.zeros(n_cause + 1, dtype=np.int8)
            for k, column in ds_cols:
                forced = _state_index(row[column], cid, column)
                if forced > n_own:
                    raise ValueError(
                        "MIN DS in the correlation tree must not exceed the "
                        f"possible DS defined for the element: component "
                        f"{cid} has {n_own} limit states, column {column} "
                        f"forces DS{forced}."
                    )
                if k > n_cause:
                    continue        # a state the causation never reaches
                lut[k] = forced
            if lut[0] != 0:
                raise ValueError(
                    f"Component {cid}: an undamaged causation component "
                    "(DS0) cannot force damage."
                )
            if np.any(np.diff(lut) < 0):
                raise ValueError(
                    f"Component {cid}: the forced damage state must not "
                    "decrease as the causation component gets more damaged."
                )
            self.causation[cid] = (cause, lut)
            self.matrix[i, 0] = cause
            self.matrix[i, 1:n_cause + 2] = lut

        # causation components before their dependents; cycles are invalid
        order, placed = [], set()
        while len(order) < len(ids):
            ready = [cid for cid in ids if cid not in placed and (
                cid not in self.causation or self.causation[cid][0] in placed)]
            if not ready:
                cycle = sorted(set(ids) - placed)
                raise ValueError(
                    f"The correlation tree has a cycle among components {cycle}."
                )
            order.extend(ready)
            placed.update(ready)
        self.order = order

    def _group_components(self):
        """Partition components into performance groups.

        Groups are formed by ``(EDP, Typology)`` pairs, giving up to three
        buckets: ``'PSD, S'``, ``'PSD, NS'`` and ``'PFA, NS'``. If explicit
        ``Performance Group`` values are present (more than one) they
        override this grouping. With ``grouping_flag=False`` all components
        form one group, keyed by the EDP of the first row.
        """
        data = self.component_data
        if "Performance Group" not in data:
            data["Performance Group"] = -1
        data["Performance Group"] = data["Performance Group"].fillna(-1)
        data["Typology"] = data["Typology"].fillna("-1")

        if not self.grouping_flag:
            self.component_groups = {data["EDP"].iloc[0]: data}
            return

        edp_groups = data.groupby(["EDP", "Typology"])
        buckets = {"PSD, S": ("psd", "s"), "PSD, NS": ("psd", "ns"),
                   "PFA, NS": ("pfa", "ns")}
        self.component_groups = {
            label: edp_groups.get_group(key)
            for label, key in buckets.items() if key in edp_groups.groups
        }
        if data["Performance Group"].nunique() > 1:
            self.component_groups = {
                group: df for group, df in data.groupby("Performance Group")
            }

    # -----------------------------------------------------------------------
    # Fragility and closed-form expectation
    # -----------------------------------------------------------------------

    def _exceedance(self, component) -> np.ndarray:
        """P(limit state >= k | EDP) of a component's own capacity, shape
        ``(n_ls, n_edp)``. Curves are kept from crossing (a higher limit
        state is never more likely than a lower one); zero at EDP = 0."""
        edp = self.edp_range
        positive = edp > 0
        poe = np.zeros((component["n_ls"], edp.size))
        for k, ls in enumerate(component["limit_states"]):
            poe[k, positive] = special.ndtr(
                np.log(edp[positive] / ls["median"]) / ls["beta"]
            )
        return np.minimum.accumulate(poe, axis=0)

    def fragility_function(self) -> dict:
        """Fragility functions of all components.

        Returns
        -------
        dict
            ``{'EDP': edp_range, 'IDs': {Component ID: {'LS1': array, ...}}}``
            where ``'LSk'`` is the probability that the component's capacity
            reaches limit state ``k`` at each EDP (before any correlation
            tree forcing). For components whose damage states are all
            sequential, limit state ``k`` is damage state ``k``.
        """
        fragilities = {"EDP": self.edp_range, "IDs": {}}
        for cid, component in self.components.items():
            poe = self._exceedance(component)
            fragilities["IDs"][cid] = {
                f"LS{k + 1}": poe[k] for k in range(component["n_ls"])
            }
        return fragilities

    def limit_state_probabilities(self) -> dict:
        """Probability of each limit state of each unit, after correlation
        tree forcing (exact).

        Returns
        -------
        dict
            ``{Component ID: array of shape (n_ls + 1, n_edp)}``; row ``k``
            is P(limit state = k | EDP), row 0 being undamaged.
        """
        exceed = {}
        for cid in self.order:
            component = self.components[cid]
            n_ls = component["n_ls"]
            own = self._exceedance(component)
            at_least = np.vstack([np.ones((1, own.shape[1])), own])
            if cid in self.causation:
                cause, lut = self.causation[cid]
                cause_at_least = exceed[cause]
                for k in range(1, n_ls + 1):
                    trigger = np.nonzero(lut >= k)[0]
                    if trigger.size:
                        forced = cause_at_least[trigger[0]]
                        at_least[k] = 1.0 - (1.0 - at_least[k]) * (1.0 - forced)
            exceed[cid] = at_least
        out = {}
        for cid in self.components:
            at_least = np.vstack([exceed[cid],
                                  np.zeros((1, self.edp_range.size))])
            out[cid] = at_least[:-1] - at_least[1:]
        return out

    def expected_component_loss(self) -> dict:
        """Expected repair cost of each component on the storey (exact,
        converted units), before the cap of a unit's repair cost at its
        replacement cost (which binds only where a damage outcome can cost
        more than replacing the unit).

        Returns
        -------
        dict
            ``{Component ID: array of shape (n_edp,)}``.
        """
        probabilities = self.limit_state_probabilities()
        out = {}
        for cid, component in self.components.items():
            ls_cost = np.array([0.0] + [ls["mean_cost"]
                                        for ls in component["limit_states"]])
            out[cid] = component["quantity"] * (ls_cost @ probabilities[cid])
        return out

    def group_replacement_cost(self, component_ids) -> float:
        """``replacement_cost`` if given, otherwise the expected replacement
        cost of the group's components (``Quantity`` times the unit
        replacement cost, summed)."""
        if self.replacement_cost is not None:
            return float(self.replacement_cost)
        value = sum(self.components[cid]["quantity"]
                    * self.components[cid]["replacement"]
                    for cid in component_ids)
        if not value > 0:
            raise ValueError(
                "The replacement cost of the group is zero: give "
                "'Replacement Cost' values or the replacement_cost argument."
            )
        return float(value)

    # -----------------------------------------------------------------------
    # Monte Carlo simulation
    # -----------------------------------------------------------------------

    @staticmethod
    def host_block(block: int, n_dependent: int, n_causation: int) -> int:
        """Causation block hosting dependent block ``block``."""
        return (block * n_causation) // n_dependent

    def apply_correlation_tree(self, states: dict) -> dict:
        """Apply the correlation tree to sampled limit states.

        Parameters
        ----------
        states : dict
            ``{Component ID: int array of shape (n_blocks, ...)}``: the limit
            state of each block of units of the component from its own
            capacity, with ``n_blocks`` the component's number of blocks
            (1 when its units are correlated).

        Returns
        -------
        dict
            The same layout with every dependent block raised to at least
            the limit state its hosting causation block forces. The input is
            not modified; without a correlation tree it is returned as is.
        """
        if not self.causation:
            return states
        final = {cid: np.array(s) for cid, s in states.items()}
        for cid in self.order:
            if cid not in self.causation:
                continue
            cause, lut = self.causation[cid]
            own, cause_states = final[cid], final[cause]
            n_d, n_c = own.shape[0], cause_states.shape[0]
            for b in range(n_d):
                host = cause_states[self.host_block(b, n_d, n_c)]
                own[b] = np.maximum(own[b], lut[host])
        return final

    def _simulate(self, rng) -> tuple:
        """Monte Carlo realisations, in converted units.

        Returns
        -------
        costs : dict
            ``{Component ID: (n_real, n_edp)}`` repair cost of the component
            on the storey.
        replacement : dict
            ``{Component ID: (n_real,)}`` replacement cost of the component
            on the storey at the realisation's unit costs.
        """
        n_real, n_edp = self.realizations, self.edp_range.size
        causes = {cause for cause, _ in self.causation.values()}
        stored, costs, replacement = {}, {}, {}
        # one random stream per component, in inventory order: a component's
        # realisations do not depend on the order components are processed in
        # (a correlation tree changes it), so models that differ only by their
        # tree share their random numbers
        children = np.random.SeedSequence(int(rng.integers(2 ** 63))).spawn(len(self.components))
        streams = {cid: np.random.default_rng(seq) for cid, seq in zip(self.components, children)}
        for cid in self.order:
            component = self.components[cid]
            n_b, n_ls = component["blocks"], component["n_ls"]
            poe = self._exceedance(component)
            stream = streams[cid]

            # unit cost of each damage state: one probability level per
            # realisation, shared by the damage states of the component (one
            # price level for repairing and replacing the same item)
            level_u = stream.random(n_real)
            unit_cost = np.array([
                _sample_unit_cost(t0, t1, fam, level_u)
                for t0, t1, fam in zip(
                    component["cost_theta0"], component["cost_theta1"],
                    component["cost_family"])
            ])
            capacity = stream.random((n_b, n_real))
            outcome_draws = {
                k: stream.random((n_b, n_real))
                for k, ls in enumerate(component["limit_states"])
                if len(ls["outcomes"]) > 1
            }
            # outcome costs per limit state, (n_outcomes, n_real)
            tables = [np.array([unit_cost[list(o)].sum(axis=0)
                                for o in ls["outcomes"]])
                      for ls in component["limit_states"]]
            # unit replacement cost at the realisation's price level, which
            # the cost of the replacement outcome sets; no unit is repaired
            # for more than it costs to replace
            k_rep, o_rep = component["replacement_outcome"]
            mean_rep = component["replacement_outcome_mean"]
            price = (tables[k_rep][o_rep] / mean_rep if mean_rep > 0
                     else np.ones(n_real))
            unit_replacement = component["replacement"] * price
            tables = [np.minimum(table, unit_replacement[None, :])
                      for table in tables]
            replacement[cid] = component["quantity"] * unit_replacement

            def step_costs(block):
                """Cost increment of each limit state over the previous one,
                (n_ls, n_real), for one block."""
                level = np.zeros((n_ls + 1, n_real))
                for k, table in enumerate(tables):
                    if k in outcome_draws:
                        choice = np.searchsorted(
                            component["limit_states"][k]["cum_p"],
                            outcome_draws[k][block], side="right")
                        level[k + 1] = table[choice, np.arange(n_real)]
                    else:
                        level[k + 1] = table[0]
                return np.diff(level, axis=0)

            forcing = self.causation.get(cid)
            if forcing is not None:
                cause, lut = forcing
                cause_states = stored[cause]
                thresholds = [np.nonzero(lut >= k)[0] for k in range(1, n_ls + 1)]
            # with one outcome per limit state every block has the same cost
            # steps, so counting the blocks in each limit state is enough
            varying = bool(outcome_draws)
            shared_steps = None if varying else step_costs(0)
            reached = None if varying else np.zeros((n_ls, n_real, n_edp), dtype=np.int32)
            total = np.zeros((n_real, n_edp))
            keep = np.empty((n_b, n_real, n_edp), dtype=np.int8) if cid in causes else None
            host_cache = (None, None)
            for b in range(n_b):
                at_least = [capacity[b][:, None] < poe[k][None, :]
                            for k in range(n_ls)]
                if forcing is not None:
                    host = self.host_block(b, n_b, cause_states.shape[0])
                    if host_cache[0] != host:
                        host_state = cause_states[host]
                        host_cache = (host, [host_state >= t[0] if t.size else None
                                             for t in thresholds])
                    for k, forced in enumerate(host_cache[1]):
                        if forced is not None:
                            at_least[k] |= forced
                if keep is not None:
                    keep[b] = np.sum(at_least, axis=0, dtype=np.int8)
                if varying:
                    steps = step_costs(b)
                    for k in range(n_ls):
                        total += at_least[k] * steps[k][:, None]
                else:
                    for k in range(n_ls):
                        reached[k] += at_least[k]
            if not varying:
                for k in range(n_ls):
                    total += reached[k] * shared_steps[k][:, None]
            costs[cid] = total * component["block_quantity"]
            if keep is not None:
                stored[cid] = keep
        return costs, replacement

    # -----------------------------------------------------------------------
    # Output
    # -----------------------------------------------------------------------

    def transform_output(self,
                         mean: np.ndarray,
                         std: np.ndarray,
                         slf_16th: np.ndarray,
                         slf_50th: np.ndarray,
                         slf_84th: np.ndarray,
                         replacement_cost: float,
                         typology: str = None) -> dict:
        """Build the SLF output record of a performance group.

        Parameters
        ----------
        mean : np.ndarray
            Mean storey loss ratio at every EDP.
        std : np.ndarray
            Standard deviation of the storey loss ratio.
        slf_16th, slf_50th, slf_84th : np.ndarray
            Empirical 16th/50th/84th percentiles of the storey loss ratio.
        replacement_cost : float
            ``replacement_cost`` if given, otherwise the expected replacement
            cost of the group's components.
        typology : str, optional
            Component type label (e.g. ``'ns'``). Default ``None``.

        Returns
        -------
        dict
            Keys ``'Directionality'``, ``'Component-type'``, ``'Storey'``,
            ``'edp'``, ``'edp_range'``, ``'slf'`` (mean loss ratio: the
            Storey Loss Function), ``'slf_std'``, ``'slf_cov'``,
            ``'slf_16th'``, ``'slf_50th'``, ``'slf_84th'`` and
            ``'replacement_cost'``.
        """
        mean = np.asarray(mean, dtype=float)
        std = np.asarray(std, dtype=float)
        cov = np.divide(std, mean, out=np.zeros_like(std), where=mean > 0)
        return {
            "Directionality":   self.directionality,
            "Component-type":   typology,
            "Storey":           self.storey,
            "edp":              self.edp,
            "edp_range":        list(self.edp_range),
            "slf":              list(mean),
            "slf_std":          list(std),
            "slf_cov":          list(cov),
            "slf_16th":         list(slf_16th),
            "slf_50th":         list(slf_50th),
            "slf_84th":         list(slf_84th),
            "replacement_cost": float(replacement_cost),
        }

    def generate(self) -> tuple:
        """Generate the Storey Loss Functions of all performance groups.

        1. Fragility of each limit state of each component; probability of
           each limit state after correlation tree forcing (exact).
        2. Monte Carlo realisations of the storey repair cost: independent
           capacities per component, or per unit where not correlated;
           explicit mutually exclusive and simultaneous damage states;
           correlation tree applied unit by unit; one unit-cost level per
           component and realisation, no unit repaired for more than its
           replacement cost.
        3. Storey loss ratio of each realisation: the repair cost divided by
           the replacement cost of the group's components at the same
           realisation's unit costs (or by ``replacement_cost`` if given).
           It lies in [0, 1], and is 1 when every unit is in its
           replacement outcome. Its mean, standard deviation and
           16th/50th/84th percentiles are the SLF.

        Returns
        -------
        out : dict
            ``{group_label: record}``, one SLF record per performance group
            (see :meth:`transform_output`). ``'slf'`` is the mean storey
            loss ratio, the curve to use in loss assessment.
        cache : dict
            Per group: ``'component'`` (inventory rows), ``'fragilities'``,
            ``'limit_state_probabilities'``, ``'expected_component_loss'``
            (exact, converted units), ``'ratio_of_expectations'`` (expected
            repair cost over expected replacement cost, exact: a check on
            ``'slf'``, which it matches to within about 0.01),
            ``'total_loss_storey'`` and ``'total_loss_storey_ratio'``
            (``(realizations, n_edp)`` arrays), ``'replacement_value'``
            (``(realizations,)``: what each realisation is divided by),
            ``'replacement_cost'``, ``'edp'`` and
            ``'empirical_16th'``/``'empirical_median'``/``'empirical_84th'``
            (the same arrays as in ``out``).
        """
        rng = np.random.default_rng(self.seed)
        fragilities = self.fragility_function()
        probabilities = self.limit_state_probabilities()
        expected = self.expected_component_loss()
        simulated, replaced = self._simulate(rng)

        out, cache = {}, {}
        for group, component_data_group in self.component_groups.items():
            if component_data_group.empty:
                continue

            if isinstance(self.typology, dict):
                typology = self.typology[group].lower()
            elif isinstance(self.typology, list) and self.typology:
                typology = self.typology[0].lower()
            else:
                typology = None

            item_ids = [int(c) for c in component_data_group["Component ID"]]
            replacement_cost = self.group_replacement_cost(item_ids)
            total = sum(simulated[cid] for cid in item_ids)
            if self.replacement_cost is None:
                value = sum(replaced[cid] for cid in item_ids)
            else:
                value = np.full(self.realizations, float(self.replacement_cost))
            ratio = np.divide(total, value[:, None], out=np.zeros_like(total),
                              where=value[:, None] > 0)
            mean = ratio.mean(axis=0)
            std = ratio.std(axis=0, ddof=1)
            slf_16th, slf_50th, slf_84th = np.percentile(ratio, [16, 50, 84], axis=0)

            group_str = str(group)
            out[group_str] = self.transform_output(
                mean, std, slf_16th, slf_50th, slf_84th, replacement_cost,
                typology,
            )
            cache[group_str] = {
                "component":                 component_data_group,
                "fragilities": {
                    "EDP": fragilities["EDP"],
                    "IDs": {cid: fragilities["IDs"][cid] for cid in item_ids},
                },
                "limit_state_probabilities": {cid: probabilities[cid] for cid in item_ids},
                "expected_component_loss":   {cid: expected[cid] for cid in item_ids},
                "ratio_of_expectations":     (sum(expected[cid] for cid in item_ids)
                                              / replacement_cost),
                "total_loss_storey":         total,
                "total_loss_storey_ratio":   ratio,
                "replacement_value":         value,
                "replacement_cost":          replacement_cost,
                "edp":                       self.edp,
                "empirical_16th":            slf_16th,
                "empirical_median":          slf_50th,
                "empirical_84th":            slf_84th,
            }

        self.cache = cache
        return out, cache
