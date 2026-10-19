"""Model Comparison companion: small-N interactive geometry of priced lattices."""

from __future__ import annotations

from dataclasses import replace

import altair as alt
import pandas as pd
import streamlit as st

from frontend.lattice_explorer_service import (
    LatticeSnapshot, VisualNode, MAX_VISUAL_STEPS, snapshot_priced_tree,
)
from frontend.pricing_service import (
    DEFAULT_THRESHOLD, MODELS, DividendInput, PricingInputError, PricingRequest,
)


COLORS = {
    "Continuation": "#8bcdf0",
    "Early exercise": "#edb970",
    "Maturity": "#a5b5c8",
    "Simplified": "#e58e8e",
}


def _plot_domain(
    snapshots: list[LatticeSnapshot], all_allocated: bool
) -> tuple[float, float]:
    """One initial Y-axis domain for BOTH models, independent of focus mode."""
    prices = [
        item.price
        for snap in snapshots
        for item in (snap.allocated_geometry if all_allocated else snap.nodes)
    ]
    if not prices:
        return (0.0, 100.0)
    low, high = min(prices), max(prices)
    pad = max((high - low) * 0.08, high * 0.02, 0.01)
    return (max(0.0, low - pad), high + pad)



def _lattice_chart_data(snapshot: LatticeSnapshot) -> tuple:
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
    links = [
        {
            "edge_id": e.edge_id,
            "step": layer,
            "stock": price,
            # Render lattice navigation without mutating the underlying tree.
            "simplified": e.simplified,
        }
        for e in snapshot.edges
        for layer, price in ((e.step, e.from_price), (e.step + 1, e.to_price))
    ]
    return nodes, links



def _lattice_edge_marks(snapshot: LatticeSnapshot, domain: tuple, links: list) -> tuple:
    """Preserve comparative graph axes and line styles."""
    x = alt.X(
        "step:Q", title="Time step",
        scale=alt.Scale(domain=[-0.2, snapshot.request.steps + 0.2]),
        axis=alt.Axis(tickMinStep=1, values=list(range(snapshot.request.steps + 1))),
    )
    y = alt.Y(
        "stock:Q", title="Underlying spot (€)",
        scale=alt.Scale(domain=list(domain), zero=False),
    )
    df = pd.DataFrame(links)
    regular_edges = alt.Chart(df[df["simplified"] == False]).mark_line(  # noqa: E712
        stroke="#61758c", opacity=0.34, strokeWidth=1.15
    ).encode(x=x, y=y, detail="edge_id:N")
    simplified_edges = alt.Chart(df[df["simplified"] == True]).mark_line(  # noqa: E712
        stroke=COLORS["Simplified"], opacity=0.85, strokeWidth=2.2,
        strokeDash=[5, 3],
    ).encode(x=x, y=y, detail="edge_id:N")
    return x, y, regular_edges, simplified_edges



def _lattice_point_marks(snapshot: LatticeSnapshot, nodes: pd.DataFrame, x: alt.X, y: alt.Y) -> tuple:
    """Keep unreached allocation and option state visual marks."""
    ghosts = pd.DataFrame([
        {"id": n.node_id, "step": n.step, "stock": n.price,
         "status": "Allocated; no positive-probability path"}
        for n in snapshot.allocated_geometry if not n.connected
    ], columns=["id", "step", "stock", "status"])
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
) -> alt.Chart:
    """Show priced positive-mass paths, optionally plus unconnected allocations."""
    nodes, links = _lattice_chart_data(snapshot)
    x, y, regular_edges, simplified_edges = _lattice_edge_marks(snapshot, domain, links)
    # All ghost points are actual Node instances observed at construction time;
    # we never draw theoretical/synthetic lattice positions as allocations.
    ghost_chart, active_points = _lattice_point_marks(snapshot, nodes, x, y)
    layers = [regular_edges, simplified_edges]
    if all_allocated:
        layers.append(ghost_chart)
    layers.append(active_points)
    return alt.layer(*layers).properties(
        height=height,
        title=snapshot.request.model,
    ).interactive()



def _inspect_connected_node(snapshot: LatticeSnapshot, prefix: str) -> None:
    """Inspect selected priced node state and transition probabilities."""
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


def _render_snapshot(
    snapshot: LatticeSnapshot,
    prefix: str,
    domain: tuple[float, float],
    *,
    all_allocated: bool,
    expanded: bool,
) -> None:
    a, b, c, d = st.columns(4)
    a.metric("Model premium", f"{snapshot.model_price:.6f} €")
    b.metric("Connected nodes", f"{snapshot.displayed_nodes:,}")
    c.metric("Actual allocations", f"{snapshot.measured_allocations:,}")
    d.metric("Simplified nodes", f"{snapshot.simplified_nodes:,}")
    st.altair_chart(
        # Use the existing chart palette and explanatory labels.
        _chart(snapshot, domain, all_allocated=all_allocated,
               height=540 if expanded else 370),
        width="stretch", key=f"{prefix}_chart",
    )
    st.caption(
        f"Unreached allocated objects: **{snapshot.unreached_allocations}**. "
        + ("Hollow grey circles = allocated but not reached. " if all_allocated else "")
        + "Dashed rose links = simplified outgoing transitions. "
        + "Only positive-probability links are drawn."
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



def _explorer_market_fields(left: object) -> tuple:
    """Collect underlying option and market inputs."""
    with left:
        st.markdown('<div class="desk-section">01 · Market & option</div>', unsafe_allow_html=True)
        aa, bb = st.columns(2)
        kind = aa.selectbox("Option", ("Call", "Put"), index=1)
        exercise = bb.selectbox("Exercise", ("European", "American"), index=1)
        aa, bb = st.columns(2)
        spot = aa.number_input("Spot · €", min_value=0.01, value=100.0, step=1.0)
        strike = bb.number_input("Strike · €", min_value=0.0, value=100.0, step=1.0)
        aa, bb = st.columns(2)
        vol = aa.number_input("Volatility · %", min_value=0.0, max_value=500.0,
                              value=20.0, step=1.0)
        rate = bb.number_input("Rate · %", min_value=-50.0, max_value=100.0,
                               value=2.0, step=0.25)
        maturity = st.number_input("Maturity · years", min_value=0.01, value=1.0, step=0.25)
    return kind, exercise, spot, strike, vol, rate, maturity



def _explorer_graph_fields(right: object) -> tuple:
    """Collect display settings and optional dividend inputs."""
    with right:
        # Keep geometric coordinate mapping separate from financial valuations.
        st.markdown('<div class="desk-section">02 · Visual comparison</div>', unsafe_allow_html=True)
        aa, bb = st.columns(2)
        first = aa.selectbox("Left model", MODELS, index=0)
        second = bb.selectbox("Right model", MODELS, index=3)
        aa, bb = st.columns(2)
        steps = aa.number_input("Visualization steps N", min_value=2,
                                max_value=MAX_VISUAL_STEPS, value=5, step=1)
        threshold = bb.selectbox("Pruning threshold", (1e-18, 1e-9, 1e-5, 1e-3), index=0,
                                 format_func=lambda x: f"{x:.0e}")
        with st.expander("Optional single cash dividend"):
            use_dividend = st.checkbox("Include cash dividend", value=False)
            aa, bb = st.columns(2)
            div_time = aa.number_input("Ex-date · years", min_value=0.01,
                                        value=0.50, step=0.05)
            div_amount = bb.number_input("Cash amount · €", min_value=0.01,
                                          value=3.0, step=0.5)
            # Use the existing chart palette and explanatory labels.
            st.caption("Binomial does not support discrete cash dividends. "
                       "A model may fail independently of the other.")
        st.caption("Use N≤10 to keep the geometry legible. A larger pruning threshold "
                   "makes simplified transitions visible but may affect the price.")
    return first, second, steps, threshold, use_dividend, div_time, div_amount



def _render_explorer_geometry(stored: list) -> None:
    """Draw shared-domain lattice plots without rebuilding models."""
    snapshots = [snap for _, snap, _ in stored if snap is not None]
    st.markdown('<div class="desk-section">03 · Priced lattice geometry</div>', unsafe_allow_html=True)
    _pruning_comparison(stored)
    aa, bb = st.columns([1, 1.25])
    view = aa.radio(
        "Workspace view", ("Side by side", "Focus left", "Focus right"),
        # Preserve node highlighting, slider state and interactive controls.
        horizontal=True, key="lattice_v23_layout",
    )
    geometry = bb.radio(
        "Display nodes", ("All allocated", "Connected only"),
        horizontal=True, key="lattice_v23_geometry",
    )
    show_all = geometry == "All allocated"
    domain = _plot_domain(snapshots, show_all)
    if view == "Side by side":
        columns = st.columns(2, gap="large")
        selected_indices = (0, 1)
    else:
        columns = [st.container()]
        selected_indices = (0,) if view == "Focus left" else (1,)
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
                )
    st.caption(
        "The Y-axis uses identical initial bounds for both models, even in focus "
        "mode. Manual zoom/pan is independent in each chart. Unreached nodes are "
        "real engine allocations, not hypothetical omitted nodes. Transitions "
        "are drawn only when their probability is positive. Early exercise is "
        "flagged where immediate exercise strictly dominates continuation."
    )



def _build_explorer_model_snapshots(
    base: PricingRequest,
    first: str,
    second: str,
) -> object:
    """Construct both independent lattices and preserve failure reporting."""
    results: list[tuple[str, LatticeSnapshot | None, str | None]] = []
    for model in (first, second):
        try:
            # Render lattice navigation without mutating the underlying tree.
            results.append((model, snapshot_priced_tree(replace(base, model=model)), None))
        except (PricingInputError, ValueError, RuntimeError, OverflowError,
                ZeroDivisionError) as exc:
            results.append((model, None, str(exc)))
    return results

def render_lattice_explorer_page() -> None:
    st.markdown('<div class="desk-kicker">Lattice explorer / 03</div>', unsafe_allow_html=True)
    st.markdown('<div class="desk-title">Interactive Lattice · Geometry & Exercise</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="desk-subtitle">Inspect real priced nodes, transition links and '
        'optimal early exercise · side-by-side architectures</div>',
        unsafe_allow_html=True,
    )

    # Keep requested lattice depth and graph visualization synchronized.
    with st.form("lattice_explorer_form"):
        left, right = st.columns([1, 1])
        kind, exercise, spot, strike, vol, rate, maturity = _explorer_market_fields(left)
        first, second, steps, threshold, use_dividend, div_time, div_amount = _explorer_graph_fields(right)
        submitted = st.form_submit_button("BUILD & COMPARE LATTICES →", type="primary", width="stretch")

    if submitted:
        divs = (DividendInput(div_time, div_amount),) if use_dividend else ()
        base = PricingRequest(
            spot=float(spot), strike=float(strike), rate=float(rate) / 100,
            volatility=float(vol) / 100, maturity=float(maturity), steps=int(steps),
            option_type=kind, exercise=exercise, pruning_threshold=threshold,
            dividends=divs,
        )
        results = _build_explorer_model_snapshots(base, first, second)
        st.session_state["lattice_explorer_result"] = results

    stored = st.session_state.get("lattice_explorer_result")
    if not stored:
        st.info("Choose two models and click **BUILD & COMPARE LATTICES**. "
                "Each chart will display the actual priced lattice geometry.")
        return
    _render_explorer_geometry(stored)
