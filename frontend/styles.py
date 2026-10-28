"""Restrained visual styling for a desk-style Streamlit pricing terminal."""

import streamlit as st


CSS = """
<style>
    .block-container {max-width: 1580px; padding-top: 1.15rem; padding-bottom: 3rem;}
    [data-testid="stSidebar"] {border-right: 1px solid #243243;}
    div[data-testid="stForm"] {border: 1px solid #2A3A4A; border-radius: 12px;
        background: #101B27; padding: 1rem 1.05rem;}
    div[data-testid="stMetric"] {border: 1px solid #29394A; background: #142130;
        border-radius: 10px; padding: 0.72rem 0.92rem;}
    div[data-testid="stMetricLabel"] {color: #A1B2C6;}
    div[data-testid="stMetricValue"] {font-variant-numeric: tabular-nums;}
    .desk-kicker {font-size: .80rem; font-weight: 700; letter-spacing: .13em;
        color: #76C4F0; text-transform: uppercase; margin: .2rem 0 .45rem;}
    .desk-title {font-size: 2rem; font-weight: 650; letter-spacing: -.04em;
        line-height: 1.13; margin: 0 0 .35rem; color: #F0F6FF;}
    .desk-subtitle {color: #A5B4C4; font-size: .98rem; margin-bottom: 1.1rem;}
    .desk-section {color: #AFBED0; letter-spacing: .11em; font-size: .79rem;
        font-weight: 750; text-transform: uppercase; margin: .6rem 0 .2rem;}
    .desk-ticket {border: 1px solid #304C60; background: linear-gradient(115deg,#142638,#172432);
        border-radius: 12px; padding: 1.2rem 1.3rem; margin: .1rem 0 1rem;}
    .desk-ticket .name {color: #A7C1D4; font-size: .79rem; font-weight: 700;
        letter-spacing: .11em; text-transform: uppercase;}
    .desk-ticket .price {color: #91D4FC; font-size: 2.7rem; font-weight: 650;
        letter-spacing: -.045em; line-height: 1.15; font-variant-numeric: tabular-nums;}
    .desk-ticket .description {color: #B8C7D8; font-size: .85rem; margin-top: .35rem;}
    div[data-testid="stFormSubmitButton"] button[kind="primary"] {
        background-color: #327DA9; border-color: #327DA9; color: white; font-weight: 650;
    }
    div[data-testid="stFormSubmitButton"] button[kind="primary"]:hover {
        background-color: #4599C8; border-color: #4599C8;
    }
    div[data-testid="stWidgetLabel"] p {font-size: .9rem; font-weight: 580;}
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {font-size: 1.6rem;}
    div[data-testid="stButton"] > button {font-weight: 600;}
    [data-testid="stDataFrame"], [data-testid="stDataEditor"] {border-radius: 8px;}
    @media(max-width: 850px) {
        .block-container {padding-top: .85rem;}
        .desk-title {font-size: 1.48rem;}
        .desk-ticket .price {font-size: 2.15rem;}
    }
</style>
"""


def apply_theme() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
