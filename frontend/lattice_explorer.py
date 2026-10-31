"""Model Comparison companion: small-N interactive geometry of priced lattices."""

from __future__ import annotations

from collections.abc import MutableMapping
from dataclasses import replace
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from frontend.diagnostics_fields import (
    render_market_fields, render_option_pair, render_reference_import,
)
from frontend.lattice_plot_sampling import (
    overview_step_indices, representative_transitions, select_chart_rows,
)
from frontend.lattice_explorer_service import (
    LatticeSnapshot, VisualNode, MAX_VISUAL_STEPS, snapshot_priced_tree,
)
from frontend.diagnostics_contract import import_saved_reference
from frontend.workspace_policy import EXPLORER_THRESHOLDS
from frontend.pricing_service import (
    DEFAULT_THRESHOLD, MODELS, DividendInput, PricingInputError, PricingRequest,
)


# Node colors also distinguish early exercise and simplified allocations.
COLORS = {
    "Continuation": "#8bcdf0",
    "Early exercise": "#edb970",
    "Maturity": "#a5b5c8",
    "Simplified": "#e58e8e",
}


def _plot_domain(
    snapshots: list[LatticeSnapshot], all_allocated: bool,
    step_window: tuple[int, int] | None = None,
) -> tuple[float, float]:
    """One initial Y-axis domain for BOTH models, independent of focus mode."""
    # The domain includes unreachable allocations only when explicitly requested.
    prices = [
        item.price
        for snap in snapshots
        for item in (snap.allocated_geometry if all_allocated else snap.nodes)
        if step_window is None or step_window[0] <= item.step <= step_window[1]
    ]
    if not prices:
        return (0.0, 100.0)
    low, high = min(prices), max(prices)
    pad = max((high - low) * 0.08, high * 0.02, 0.01)
    return (max(0.0, low - pad), high + pad)



def _lattice_chart_data(snapshot: LatticeSnapshot, *, include_links: bool = True,
    step_window: tuple[int, int] | None = None,
    max_links: int | None = None,
) -> tuple:
    """Map actual graph objects to plotted node and edge data."""
    nodes = pd.DataFrame([
        {
            "id": n.node_id, "step": n.step, "stock": n.price,
            "value": n.option_value, "mass": n.reach_probability,
            "state": "Simplified" if n.simplified else n.state,
            "up": n.up_probability, "mid": n.mid_probability,
            "down": n.down_probability,
        }
        for n in snapshot.nodes
    ])
    # Link drawing is a view filter: original priced edges remain immutable.
    visible_edges = snapshot.edges if include_links else ()
    if include_links and max_links is not None:
        visible_edges = representative_transitions(
            visible_edges, step_window=step_window, max_links=max_links,
        )
    links = [
        {
            "edge_id": e.edge_id,
            "step": layer,
            "stock": price,
            # Render lattice navigation without mutating the underlying tree.
            "simplified": e.simplified,
        }
        for e in visible_edges
        if step_window is None or step_window[0] <= e.step < step_window[1]
        for layer, price in ((e.step, e.from_price), (e.step + 1, e.to_price))
    ]
    return nodes, links



def _lattice_axes(
    snapshot: LatticeSnapshot, domain: tuple,
    step_window: tuple[int, int] | None = None,
) -> tuple:
    """Sensible tick spacing at large N; shared by node and edge views."""
    n = snapshot.request.steps
    low, high = step_window if step_window is not None else (0, n)
    stride = max(1, (high - low + 11) // 12)
    ticks = sorted(set([*range(low, high + 1, stride), high]))
    x = alt.X(
        "step:Q", title="Time step",
        scale=alt.Scale(domain=[low - 0.2, high + 0.2]),
        axis=alt.Axis(tickMinStep=1, values=ticks),
    )
    y = alt.Y(
        "stock:Q", title="Underlying spot (€)",
        scale=alt.Scale(domain=list(domain), zero=False),
    )
    return x, y


def _lattice_edge_marks(
    snapshot: LatticeSnapshot, domain: tuple, links: list,
    step_window: tuple[int, int] | None = None,
) -> tuple:
    """Preserve exact positive-probability edge identities when requested."""
    x, y = _lattice_axes(snapshot, domain, step_window)
    frame = pd.DataFrame(links)
    regular_edges = alt.Chart(frame[frame["simplified"] == False]).mark_line(  # noqa: E712
        stroke="#61758c", opacity=0.34, strokeWidth=1.15
    ).encode(x=x, y=y, detail="edge_id:N")
    # Display only genuine priced nodes and edges without altering the lattice.
    simplified_edges = alt.Chart(frame[frame["simplified"] == True]).mark_line(  # noqa: E712
        stroke=COLORS["Simplified"], opacity=0.85, strokeWidth=2.2,
        strokeDash=[5, 3],
    ).encode(x=x, y=y, detail="edge_id:N")
    return x, y, regular_edges, simplified_edges



def _lattice_point_marks(snapshot: LatticeSnapshot, nodes: pd.DataFrame, x: alt.X, y: alt.Y,
    *, step_window: tuple[int, int] | None = None, overview: bool = False,
) -> tuple:
    """Keep unreached allocation and option state visual marks."""
    # Ghost marks correspond to actually allocated nodes with no positive-mass path.
    ghosts = pd.DataFrame([
        {"id": n.node_id, "step": n.step, "stock": n.price,
         "status": "Allocated; no positive-probability path"}
        for n in snapshot.allocated_geometry if not n.connected
    ], columns=["id", "step", "stock", "status"])
    if overview:
        retained = overview_step_indices(
            snapshot.request.steps,
            event_times=tuple(d.time for d in snapshot.request.dividends),
            maturity=snapshot.request.maturity,
        )
        ghosts = ghosts.loc[ghosts["step"].isin(retained)]
    ghosts = select_chart_rows(
        ghosts, step_window=step_window, overview=overview,
        per_step=8 if overview else 10,
    )
    # Hollow circles identify unreached allocations, never hypothetical nodes.
    ghost_chart = alt.Chart(ghosts).mark_point(
        filled=False, stroke="#9ba9ba", strokeWidth=1.4,
        opacity=0.75, size=82,
    ).encode(
        x=x, y=y,
        tooltip=[
            alt.Tooltip("id:N", title="Allocated node"),
            alt.Tooltip("step:Q", title="Step"),
            alt.Tooltip("stock:Q", title="Spot (€)", format=".6f"),
            alt.Tooltip("status:N", title="Status"),
        ],
    )
    # Preserve node highlighting, slider state and interactive controls.
    active_points = alt.Chart(nodes).mark_circle(
        size=77, stroke="#06121d", strokeWidth=1.0,
    ).encode(
        x=x, y=y,
        color=alt.Color(
            "state:N", title="Priced node state",
            scale=alt.Scale(domain=list(COLORS), range=list(COLORS.values())),
        ),
        tooltip=[
            alt.Tooltip("id:N", title="Node"),
            alt.Tooltip("step:Q", title="Step"),
            alt.Tooltip("stock:Q", title="Spot (€)", format=".6f"),
            alt.Tooltip("value:Q", title="Option value (€)", format=".6f"),
            alt.Tooltip("mass:Q", title="Reach probability", format=".8f"),
            alt.Tooltip("state:N", title="State"),
            alt.Tooltip("up:Q", title="P(up)", format=".6f"),
            alt.Tooltip("mid:Q", title="P(mid)", format=".6f"),
            # Keep geometric coordinate mapping separate from financial valuations.
            alt.Tooltip("down:Q", title="P(down)", format=".6f"),
        ],
    )
    return ghost_chart, active_points


def _chart(
    snapshot: LatticeSnapshot,
    domain: tuple[float, float],
    *,
    all_allocated: bool,
    height: int,
    show_links: bool = True,
    step_window: tuple[int, int] | None = None,
    overview: bool = False,
    max_links: int | None = None,
) -> alt.Chart:
    """Sparse global chart or complete local priced nodes; never reprice."""
    if overview and show_links:
        raise ValueError("Full links require the detailed time-window view.")
    # Display only genuine priced nodes and edges without altering the lattice.
    nodes, links = _lattice_chart_data(
        snapshot, include_links=show_links, step_window=step_window,
        max_links=max_links,
    )
    if overview:
        # Keep overview responsive: representative layers and node states only.
        # Full geometry remains unchanged in the priced LatticeSnapshot.
        retained = overview_step_indices(
            snapshot.request.steps,
            event_times=tuple(d.time for d in snapshot.request.dividends),
            maturity=snapshot.request.maturity,
        )
        nodes = nodes.loc[nodes["step"].isin(retained)]
    nodes = select_chart_rows(
        nodes, step_window=step_window, overview=overview,
        per_step=8 if overview else 10,
    )
    # Edge visibility never changes the underlying priced nodes or transition set.
    if show_links:
        x, y, normal, simplified = _lattice_edge_marks(
            snapshot, domain, links, step_window,
        )
        layers = [normal, simplified]
    else:
        x, y = _lattice_axes(snapshot, domain, step_window)
        layers = []
    ghost_chart, active_points = _lattice_point_marks(
        snapshot, nodes, x, y, step_window=step_window, overview=overview,
    )
    if all_allocated:
        layers.append(ghost_chart)
    layers.append(active_points)
    return alt.layer(*layers).properties(
        height=height,
        title=snapshot.request.model,
    ).interactive()



def _inspect_connected_node(snapshot: LatticeSnapshot, prefix: str) -> None:
    """Inspect selected priced node state and transition probabilities."""
    # Display only genuine priced nodes and edges without altering the lattice.
    st.caption("Inspect the engine-priced object and its transition probabilities.")
    a, b = st.columns([1, 2])
    step = a.selectbox(
        "Layer", range(snapshot.request.steps + 1), key=f"{prefix}_step",
        format_func=lambda v: f"Step {v} · t={v * snapshot.request.maturity / snapshot.request.steps:.3f}Y",
    )
    subset = [n for n in snapshot.nodes if n.step == step]
    # Preserve node highlighting, slider state and interactive controls.
    chosen = b.selectbox(
        "Node (high to low spot)", [n.node_id for n in subset],
        key=f"{prefix}_node",
        format_func=lambda ident: next(
            f"{ident} · S={node.price:.4f} €" for node in subset if node.node_id == ident
        ),
    )
    node = next(n for n in subset if n.node_id == chosen)
    cols = st.columns(3)
    cols[0].metric("Spot (€)", f"{node.price:,.4f}")
    cols[1].metric("Option value (€)", f"{node.option_value:,.5f}")
    cols[2].metric("Reach probability", f"{node.reach_probability:.4%}")
    st.dataframe(pd.DataFrame([{
        "State": node.state,
        "Simplified transition": "Yes" if node.simplified else "No",
        "P(up)": node.up_probability, "P(mid)": node.mid_probability,
        "P(down)": node.down_probability,
    # Keep geometric coordinate mapping separate from financial valuations.
    }]), hide_index=True, width="stretch")


def _display_node_details(snapshot: LatticeSnapshot, prefix: str) -> None:
    inactive = [n for n in snapshot.allocated_geometry if not n.connected]
    selection = "Priced / connected"
    if inactive:
        # Use the existing chart palette and explanatory labels.
        selection = st.radio(
            "Inspect", ("Priced / connected", "Allocated, not reached"),
            horizontal=True, key=f"{prefix}_node_scope",
        )
    if selection == "Allocated, not reached":
        st.caption(
            "These are real allocated objects, but they have no path with "
            "positive probability from the root. The recursive engine may "
            "still visit some through zero-weight links; their option value "
            "is not a probability-weighted contribution."
        )
        steps = sorted({n.step for n in inactive})
        a, b = st.columns([1, 2])
        step = a.selectbox("Layer", steps, key=f"{prefix}_inactive_step")
        items = [n for n in inactive if n.step == step]
        chosen = b.selectbox(
            "Allocated node", range(len(items)),
            # Render lattice navigation without mutating the underlying tree.
            key=f"{prefix}_inactive_node",
            format_func=lambda i: f"{items[i].node_id} · S={items[i].price:.4f} €",
        )
        node = items[chosen]
        a.metric("Spot (€)", f"{node.price:,.4f}")
        b.metric("Option value (€)",
                 f"{node.option_value:,.5f}" if node.option_value is not None
                 else "Not computed")
        return

    _inspect_connected_node(snapshot, prefix)


def focus_lattice_snapshot(
    snapshot: LatticeSnapshot,
    step_window: tuple[int, int],
    spot_band: tuple[float, float],
) -> LatticeSnapshot:
    """Induced priced subgraph inside an exact time/spot rectangle.

    Unlike link sampling, this preserves ALL real edges with BOTH endpoints
    visible. No interpolation, independent edge sampling or price adjustments.
    The original snapshot is immutable and remains available to Node inspector.
    """
    # Display only genuine priced nodes and edges without altering the lattice.
    start, end = step_window
    low, high = spot_band
    if start < 0 or end < start or end > snapshot.request.steps:
        raise ValueError("Invalid lattice time window.")
    if not (0 <= low < high):
        raise ValueError("Invalid spot focus interval.")

    # Induce the exact subgraph: both edge endpoints must remain visible.
    nodes = tuple(node for node in snapshot.nodes
                  if start <= node.step <= end and low <= node.price <= high)
    visible_ids = {node.node_id for node in nodes}
    edges = tuple(edge for edge in snapshot.edges
                  if start <= edge.step < end
                  and edge.from_id in visible_ids and edge.to_id in visible_ids)
    allocated = tuple(node for node in snapshot.allocated_geometry
                      if start <= node.step <= end and low <= node.price <= high)
    return replace(snapshot, nodes=nodes, edges=edges,
                   allocated_geometry=allocated)


def _render_lattice_explanations(
    snapshot: LatticeSnapshot, all_allocated: bool, show_links: bool,
    overview: bool, spot_band: tuple[float, float] | None,
) -> None:
    """Keep graph disclosures and dividend exercise qualifications together."""
    if spot_band is not None:
        st.caption("Focused lattice · every node and connection inside the "
                   f"spot interval [{spot_band[0]:.2f}, {spot_band[1]:.2f}] €. "
                   "No sampled or invented paths. Node inspector retains "
                   "access to the complete tree.")
    # Display only genuine priced nodes and edges without altering the lattice.
    if overview:
        st.caption("Fast overview: sampled time layers and representative "
                   "priced nodes, retaining the extreme spots and exercise "
                   "states. Detailed window shows every local priced node.")
    st.caption(
        f"Unreached allocated objects: **{snapshot.unreached_allocations}**. "
        + ("Hollow grey circles = allocated but not reached. " if all_allocated else "")
        + ("Dashed rose links = simplified outgoing transitions. "
           "Only positive-probability links are drawn. " if show_links else
           "Individual transition links are hidden for responsiveness; "
           "all connected node values remain visible. ")
    )
    if snapshot.request.exercise == "American" and snapshot.request.dividends:
        st.caption(
            "Exercise markers represent decisions at lattice nodes only. "
            "Pre/post dividend-event exercise is reflected in model values, "
            "but not drawn as an exercise node."
        )


# The renderer reuses a priced snapshot; display focus never changes valuation.
def _render_snapshot(
    snapshot: LatticeSnapshot,
    prefix: str,
    domain: tuple[float, float],
    *,
    all_allocated: bool,
    expanded: bool,
    show_links: bool = True,
    step_window: tuple[int, int] | None = None,
    overview: bool = False,
    max_links: int | None = None,
    spot_band: tuple[float, float] | None = None,
) -> None:
    # Aggregate diagnostics describe the complete graph, before chart sampling.
    a, b, c, d = st.columns(4)
    a.metric("Model premium", f"{snapshot.model_price:.6f} €")
    b.metric("Connected nodes", f"{snapshot.displayed_nodes:,}")
    c.metric("Actual allocations", f"{snapshot.measured_allocations:,}")
    d.metric("Simplified nodes", f"{snapshot.simplified_nodes:,}")
    # Override Altair's 5,000-row guard only while Streamlit serializes
    # this explicit, user-requested full-node lattice visualization.
    # Local focus operates on a copy of stored graph geometry, never on pricing.
    chart_snapshot = (
        focus_lattice_snapshot(snapshot, step_window, spot_band)
        if step_window is not None and spot_band is not None else snapshot
    )
    # Even deterministic zero-volatility trees can have thousands of nodes
    # inside a very small spot window. Ask for a narrower view rather than lag.
    if not chart_snapshot.nodes:
        st.info("No priced nodes in this spot interval. Move Spot focus "
                "or widen the visible spot range.")
    elif (
        spot_band is not None
        and (len(chart_snapshot.nodes) > 500
             or len(chart_snapshot.edges) > 1400)
    ):
        st.warning("This local view is still too dense to render smoothly. "
                   "Narrow the spot range or the time window. The full lattice "
                   "and its node inspector remain available below.")
    else:
        with alt.data_transformers.enable("default", max_rows=None):
            st.altair_chart(
                _chart(chart_snapshot, domain, all_allocated=all_allocated,
                       height=540 if expanded else 370,
                       show_links=show_links and bool(chart_snapshot.edges),
                       step_window=step_window, overview=overview,
                       max_links=max_links),
                width="stretch", key=f"{prefix}_chart",
            )
    # Display only genuine priced nodes and edges without altering the lattice.
    _render_lattice_explanations(
        snapshot, all_allocated, show_links, overview, spot_band,
    )
    with st.expander("Node inspector", expanded=False):
        _display_node_details(snapshot, prefix)


def _pruning_comparison(
    stored: list[tuple[str, LatticeSnapshot | None, str | None]],
) -> None:
    """Show savings only when comparing the SAME trinomial family."""
    available = {
        name: snap for name, snap, error in stored if snap is not None and error is None
    }
    reference = available.get("Trinomial · Branch pruning") or available.get("Trinomial · Full")
    optimized = available.get("Trinomial · Allocation pruning")
    if reference is None or optimized is None:
        return
    ref_count = reference.measured_allocations
    opt_count = optimized.measured_allocations
    # Display only genuine priced nodes and edges without altering the lattice.
    if not ref_count:
        return
    diff = ref_count - opt_count
    price_gap = abs(reference.model_price - optimized.model_price)
    a, b, c = st.columns(3)
    a.metric("Reference allocations", f"{ref_count:,}")
    b.metric("Optimized allocations", f"{opt_count:,}")
    c.metric("Saved vs reference", f"{diff / ref_count:.2%}",
             # Render lattice navigation without mutating the underlying tree.
             help="Measured from actual created Node objects, not theoretical bounds.")
    if price_gap > 1e-9:
        st.warning(
            f"Numerical prices differ by {price_gap:.6g} €. "
            "An allocation reduction does not imply price preservation at this threshold."
        )



def apply_reference_to_explorer(
    state: MutableMapping[str, Any],
) -> tuple[bool, tuple[str, ...]]:
    """Explicit exact import, preserving lattice-model and view choices."""
    return import_saved_reference(state, "Explore lattice")


def explorer_dividend_defaults(
    seed: PricingRequest | None,
) -> tuple[bool, float, float]:
    """Match the existing single-dividend editor without discarding cashflows."""
    # Display only genuine priced nodes and edges without altering the lattice.
    if seed is None or not seed.dividends:
        return False, 0.5, 3.0
    if len(seed.dividends) != 1:
        raise ValueError("Explorer editor supports at most one cash dividend.")
    dividend = seed.dividends[0]
    return True, dividend.time, dividend.amount


def _explorer_control_key(name: str) -> str:
    """Rotate only contract widgets when the user explicitly imports."""
    generation = st.session_state.get("explorer_import_nonce", 0)
    return f"lx_{name}_{generation}" if generation else f"lx_{name}"


def _explorer_default(field: str, fallback: Any) -> Any:
    seed = st.session_state.get("explorer_reference_seed")
    return getattr(seed, field) if isinstance(seed, PricingRequest) else fallback


def _render_explorer_import() -> None:
    """Explicit ticket prefill; lattice builds remain button-triggered."""
    # Display only genuine priced nodes and edges without altering the lattice.
    render_reference_import("Explore lattice")


def _explorer_market_fields(left: object) -> tuple:
    """Shared option/market inputs, with the explorer's existing Streamlit keys."""
    with left:
        st.markdown('<div class="desk-section">01 · Market & option</div>',
                    unsafe_allow_html=True)
        kind, exercise = render_option_pair(
            _explorer_control_key, _explorer_default,
            widget_names=("kind", "exercise"),
            default_option="Put", default_exercise="American",
        )
        spot, strike, vol, rate, maturity = render_market_fields(
            _explorer_control_key, _explorer_default,
            widget_names=("spot", "strike", "volatility", "rate", "maturity"),
            rate_label="Rate · %",
        )
    return kind, exercise, spot, strike, vol, rate, maturity


def _explorer_graph_fields(right: object) -> tuple:
    """Collect display settings and optional dividend inputs."""
    # Display only genuine priced nodes and edges without altering the lattice.
    with right:
        # Keep geometric coordinate mapping separate from financial valuations.
        st.markdown('<div class="desk-section">02 · Visual comparison</div>', unsafe_allow_html=True)
        aa, bb = st.columns(2)
        first = aa.selectbox("Left model", MODELS, index=0, key="lx_left_model")
        second = bb.selectbox("Right model", MODELS, index=3, key="lx_right_model")
        aa, bb = st.columns(2)
        steps = aa.number_input("Visualization steps N", min_value=2,
                                max_value=MAX_VISUAL_STEPS,
                                value=int(_explorer_default("steps", 5)), step=1,
                                key=_explorer_control_key("steps"))
        threshold_choices = EXPLORER_THRESHOLDS
        threshold = bb.selectbox("Pruning threshold", threshold_choices,
                                 index=threshold_choices.index(_explorer_default(
                                     "pruning_threshold", DEFAULT_THRESHOLD)),
                                 format_func=lambda x: f"{x:.0e}",
                                 key=_explorer_control_key("threshold"))
        with st.expander("Optional single cash dividend"):
            seed = st.session_state.get("explorer_reference_seed")
            div_present, div_default_time, div_default_amount = explorer_dividend_defaults(
                seed if isinstance(seed, PricingRequest) else None)
            use_dividend = st.checkbox("Include cash dividend", value=div_present,
                                       key=_explorer_control_key("use_dividend"))
            # Cash-dividend timing and amount are independent of graph sampling.
            aa, bb = st.columns(2)
            div_time = aa.number_input("Ex-date · years", min_value=0.01,
                                        value=div_default_time, step=0.05,
                                        key=_explorer_control_key("div_time"))
            div_amount = bb.number_input("Cash amount · €", min_value=0.01,
                                          value=div_default_amount, step=0.5,
                                          key=_explorer_control_key("div_amount"))
            # Use the existing chart palette and explanatory labels.
            st.caption("Binomial does not support discrete cash dividends. "
                       "A model may fail independently of the other.")
        st.caption("N<20: complete lattice by default. From N=20, "
                   "Fast overview displays a sample; Detailed time window "
                   "preserves every local node. Pruning can affect price.")
    return first, second, steps, threshold, use_dividend, div_time, div_amount



def _has_ghost_allocations(snapshots: list[LatticeSnapshot]) -> bool:
    """Whether All allocated differs from Connected only in this run."""
    return any(snapshot.unreached_allocations > 0 for snapshot in snapshots)


def _explorer_display_controls(snapshots: list[LatticeSnapshot]) -> tuple[str, bool]:
    """Choose focus and allocated-node visibility without rebuilding lattices."""
    aa, bb = st.columns([1, 1.25])
    view = aa.radio(
        "Workspace view", ("Side by side", "Focus left", "Focus right"),
        # Preserve node highlighting, slider state and interactive controls.
        horizontal=True, key="lattice_v23_layout",
    )
    if _has_ghost_allocations(snapshots):
        geometry = bb.radio(
            "Display nodes", ("All allocated", "Connected only"),
            horizontal=True, key="lattice_v23_geometry",
        )
        show_all = geometry == "All allocated"
    else:
        bb.caption("Display nodes: all allocated nodes are already connected.")
        show_all = False
    return view, show_all


def _explorer_local_window(snapshots: list[LatticeSnapshot]) -> tuple:
    """Choose the exact time/spot corridor and genuine visible transitions."""
    max_steps = min(snap.request.steps for snap in snapshots)
    center = st.slider(
        "Inspect around time step", min_value=0, max_value=max_steps,
        value=max_steps // 2, step=1,
        key="lattice_v25_center_step",
    )
    radius = st.select_slider(
        "Window half-width (steps)", options=(2, 3, 4, 5),
        value=3, key="lattice_v25_window_radius",
    )
    step_window = (max(0, center - radius), min(max_steps, center + radius))
    left_focus, right_focus = st.columns(2)
    focus_spot = left_focus.number_input(
        "Spot focus (€)", min_value=0.01,
        value=float(snapshots[0].request.spot), step=5.0,
        format="%.2f", key="lattice_v28_focus_spot",
        help="Move the local lattice view up or down the spot axis.",
    )
    width_pct = right_focus.select_slider(
        "Visible spot range (±%)", options=(5, 10, 15, 20, 25),
        value=15, key="lattice_v28_spot_radius",
        help="Only real nodes and links inside this spot interval are drawn.",
    )
    spot_band = (
        # Display only genuine priced nodes and edges without altering the lattice.
        max(0.0, focus_spot * (1 - width_pct / 100)),
        focus_spot * (1 + width_pct / 100),
    )
    show_links = st.checkbox(
        "Show all visible transitions (exact)",
        value=False, key="lattice_v25_window_links",
        help="Draws every actual edge connecting nodes inside "
             "the chosen time and spot rectangle, with no link sampling.",
    )
    st.caption("Detailed window: original nodes and complete "
               "transitions in the selected time AND spot intervals. "
               "Move Spot focus to explore other regions; Node inspector "
               "still provides the entire priced tree.")
    return step_window, spot_band, show_links


def _explorer_detail_controls(snapshots: list[LatticeSnapshot]) -> tuple:
    """Configure fast overview or exact local window; preserve widget order."""
    large_tree = any(snap.request.steps >= 20 for snap in snapshots)
    step_window = None
    # Display only genuine priced nodes and edges without altering the lattice.
    spot_band = None
    overview = False
    show_links = True
    if large_tree:
        mode = st.radio(
            "Lattice detail", ("Fast overview", "Detailed time window"),
            horizontal=True, key="lattice_v25_detail_mode",
            help="Overview renders representative nodes, not every node. "
                 "Detailed time window shows every priced node locally.",
        )
        overview = mode == "Fast overview"
        if overview:
            show_links = False
            st.caption("Fast overview: a sampled timeline (about 35 of 101 "
                       "dates at N=100), up to 8 representative nodes per "
                       "displayed date. Price extremes, exercise states and "
                       "dividend-adjacent layers are retained. Every value remains "
                       "available in Node inspector or Detailed time window.")
        else:
            step_window, spot_band, show_links = _explorer_local_window(snapshots)
    return step_window, spot_band, overview, show_links


def _explorer_shared_domain(
    snapshots: list[LatticeSnapshot], show_all: bool,
    step_window: tuple[int, int] | None, spot_band: tuple[float, float] | None,
) -> tuple[float, float]:
    """Use the same price-axis bounds across both comparison panels."""
    domain = _plot_domain(snapshots, show_all, step_window)
    if spot_band is not None:
        lo, hi = spot_band
        pad = (hi - lo) * 0.04
        domain = (max(0.0, lo - pad), hi + pad)
    return domain


def _explorer_plot_panels(
    stored: list, snapshots: list[LatticeSnapshot], view: str,
    show_all: bool, step_window: tuple | None, spot_band: tuple | None,
    overview: bool, show_links: bool,
) -> None:
    """Display actual stored lattice snapshots on one comparable domain."""
    max_links = None  # Display every genuine edge inside the local corridor.
    domain = _explorer_shared_domain(
        # Display only genuine priced nodes and edges without altering the lattice.
        snapshots, show_all, step_window, spot_band,
    )
    # Both panels share one price axis, including when the user focuses a model.
    if view == "Side by side":
        columns = st.columns(2, gap="large")
        selected_indices = (0, 1)
    else:
        columns = [st.container()]
        selected_indices = (0,) if view == "Focus left" else (1,)
    # An unavailable model remains visibly marked as an error, not a zero price.
    for target, index in zip(columns, selected_indices):
        model, snap, error = stored[index]
        with target:
            # Keep geometric coordinate mapping separate from financial valuations.
            st.markdown(f"**{model}**")
            if error is not None or snap is None:
                st.error(f"{model}: {error}")
            else:
                _render_snapshot(
                    snap, f"lattice_{index}", domain,
                    all_allocated=show_all, expanded=view != "Side by side",
                    show_links=show_links, step_window=step_window,
                    overview=overview, max_links=max_links,
                    spot_band=spot_band,
                )
    st.caption(
        "The Y-axis uses identical initial bounds for both models, even in focus "
        "mode. Manual zoom/pan is independent in each chart. Unreached nodes are "
        "real engine allocations, not hypothetical omitted nodes. Transitions "
        "are drawn only when their probability is positive. Early exercise is "
        "flagged where immediate exercise strictly dominates continuation."
    )


def _render_explorer_geometry(stored: list) -> None:
    """Draw shared-domain lattice plots without rebuilding models."""
    # Display only genuine priced nodes and edges without altering the lattice.
    snapshots = [snap for _, snap, _ in stored if snap is not None]
    st.markdown('<div class="desk-section">03 · Priced lattice geometry</div>', unsafe_allow_html=True)
    st.caption(
        "Explore: wheel/trackpad to zoom · drag to pan · hover for node data · "
        "chart full-screen icon (top right) to enlarge. Focus left/right expands one model."
    )
    _pruning_comparison(stored)
    view, show_all = _explorer_display_controls(snapshots)
    step_window, spot_band, overview, show_links = _explorer_detail_controls(snapshots)
    _explorer_plot_panels(stored, snapshots, view, show_all, step_window, spot_band, overview, show_links)



def _build_explorer_model_snapshots(
    base: PricingRequest,
    first: str,
    second: str,
) -> list[tuple[str, LatticeSnapshot | None, str | None]]:
    """Construct both independent lattices and preserve failure reporting."""
    results: list[tuple[str, LatticeSnapshot | None, str | None]] = []
    # Display only genuine priced nodes and edges without altering the lattice.
    for model in (first, second):
        try:
            # Render lattice navigation without mutating the underlying tree.
            results.append((model, snapshot_priced_tree(replace(base, model=model)), None))
        except (PricingInputError, ValueError, RuntimeError, OverflowError,
                ZeroDivisionError) as exc:
            results.append((model, None, str(exc)))
    return results

def _render_explorer_heading() -> None:
    """Preserve the standalone explorer header and its existing visual hierarchy."""
    st.markdown('<div class="desk-kicker">Lattice explorer / 03</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="desk-title">Interactive Lattice · Geometry & Exercise</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="desk-subtitle">Inspect real priced nodes, transition links and '
        'optimal early exercise · side-by-side architectures</div>',
        unsafe_allow_html=True,
    )


def _store_explorer_snapshots(base: PricingRequest, first: str, second: str) -> None:
    """Store both model results without overwriting the last valid reference."""
    # Display only genuine priced nodes and edges without altering the lattice.
    results = _build_explorer_model_snapshots(base, first, second)
    st.session_state["lattice_explorer_result"] = results
    if any(snapshot is not None for _, snapshot, _ in results):
        st.session_state["diagnostics_last_contract"] = base
        st.session_state["diagnostics_last_source"] = "Explore lattice"


def render_lattice_explorer_page(*, embedded: bool = False) -> None:
    """Render the small-grid explorer within or outside Model Diagnostics."""
    if not embedded:
        _render_explorer_heading()

    _render_explorer_import()

    # Keep requested lattice depth and graph visualization synchronized.
    with st.form("lattice_explorer_form"):
        left, right = st.columns([1, 1])
        kind, exercise, spot, strike, vol, rate, maturity = _explorer_market_fields(left)
        first, second, steps, threshold, use_dividend, div_time, div_amount = _explorer_graph_fields(right)
        submitted = st.form_submit_button("BUILD & COMPARE LATTICES →", type="primary", width="stretch")

    if submitted:
        # A submit constructs two fresh model snapshots for one immutable contract.
        divs = (DividendInput(div_time, div_amount),) if use_dividend else ()
        base = PricingRequest(
            spot=float(spot), strike=float(strike), rate=float(rate) / 100,
            volatility=float(vol) / 100, maturity=float(maturity), steps=int(steps),
            option_type=kind, exercise=exercise, pruning_threshold=threshold,
            dividends=divs,
        )
        _store_explorer_snapshots(base, first, second)

    stored = st.session_state.get("lattice_explorer_result")
    if not stored:
        st.info("Choose two models and click **BUILD & COMPARE LATTICES**. "
                "Each chart will display the actual priced lattice geometry.")
        return
    _render_explorer_geometry(stored)
