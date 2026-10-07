import streamlit as st
import pandas as pd
import textwrap
from src.ui.utils import run_prediction, load_data

def render_product_page():
    st.markdown('<div class="main-header">🎯 Pre-Toss Match Predictor</div>', unsafe_allow_html=True)
    st.markdown("Select a match to predict the Dream11 Best XI and C/VC. This prediction uses strictly pre-toss features.")
    
    pms_df, feats_df, dels_df = load_data()
    all_teams = sorted(list(dels_df['batting_team'].dropna().unique()))
    
    # Demo match defaults
    default_date = pd.to_datetime('2024-07-01')
    idx_a = all_teams.index("Malawi") if "Malawi" in all_teams else 0
    idx_b = all_teams.index("Kenya") if "Kenya" in all_teams else 1
    
    col1, col2, col3 = st.columns(3)
    with col1:
        date_input = st.date_input("Match Date", default_date)
    with col2:
        team_a = st.selectbox("Team 1", all_teams, index=idx_a)
    with col3:
        team_b = st.selectbox("Team 2", all_teams, index=idx_b)
        
    st.markdown("---")
    
    use_genai = st.toggle("🧠 Explain with AI (Natural Language)", value=False)
    if st.button("🚀 Run Pre-Toss Prediction", type="primary"):
        with st.spinner("Generating prediction and explanations..."):
            try:
                pool, dream_team, latencies = run_prediction(
                    str(date_input), team_a, team_b, explain=True, use_genai=use_genai
                )
                
                st.markdown("### 🏆 Recommended Dream11 Best XI")
                
                k1, k2, k3 = st.columns(3)
                k1.markdown(f'<div class="kpi-card"><div class="kpi-val" style="color:#10b981;">{dream_team["total_predicted_score"]:.1f}</div><div class="kpi-lbl">Expected Team FP</div></div>', unsafe_allow_html=True)
                k2.markdown(f'<div class="kpi-card"><div class="kpi-val" style="color:#fbbf24;">{dream_team["captain"]}</div><div class="kpi-lbl">Captain (2x)</div></div>', unsafe_allow_html=True)
                k3.markdown(f'<div class="kpi-card"><div class="kpi-val" style="color:#38bdf8;">{dream_team["vice_captain"]}</div><div class="kpi-lbl">Vice Captain (1.5x)</div></div>', unsafe_allow_html=True)
                
                st.markdown('<div class="pitch-container"><div class="pitch-header">🏆 DREAM11 RECOMMENDED XI FORMATION</div>', unsafe_allow_html=True)
                
                roles_order = [("WK", "🧤 WICKET-KEEPERS"), ("BAT", "🏏 BATSMEN"), ("AR", "⚔️ ALL-ROUNDERS"), ("BOWL", "🎯 BOWLERS")]
                
                selected_players = dream_team['team_df']
                explanations = {exp['player']: exp for exp in dream_team['explanations']}
                
                for r_code, r_title in roles_order:
                    sub = selected_players[selected_players["role"] == r_code]
                    if not sub.empty:
                        st.markdown(f'<div style="color: #6ee7b7; font-weight: 700; font-size: 13px; margin: 12px 0 8px 0; letter-spacing: 1px;">{r_title} ({len(sub)})</div>', unsafe_allow_html=True)
                        cols = st.columns(min(len(sub), 4))
                        
                        for idx, (_, p) in enumerate(sub.iterrows()):
                            player = p['player']
                            exp = explanations.get(player, {})
                            role = exp.get('role', 'Unknown')
                            team = exp.get('team', 'Unknown')
                            p_play = exp.get('p_play', 0.0)
                            exp_fp = exp.get('expected_fp', 0.0)
                            
                            is_c = (player == dream_team['captain'])
                            is_vc = (player == dream_team['vice_captain'])
                            
                            c_class = "captain" if is_c else "vice-captain" if is_vc else ""
                            badge_html = ""
                            if is_c:
                                badge_html = '<span class="badge-c">C (2x)</span>'
                            elif is_vc:
                                badge_html = '<span class="badge-vc">VC (1.5x)</span>'
                            
                            t_color = "#38bdf8" if team == team_a else "#f43f5e"
                            
                            card_html = textwrap.dedent(f"""
                            <div class="player-card {c_class}">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                                    <span class="badge-role">{role}</span>
                                    {badge_html}
                                </div>
                                <div style="font-size: 15px; font-weight: 700; color: #ffffff; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{player}</div>
                                <div style="font-size: 11px; color: {t_color}; font-weight: 700; margin-bottom: 6px;">{team}</div>
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <span style="font-size: 11px; color: #94a3b8;">P(play): <b>{p_play:.0%}</b></span>
                                    <span class="badge-pts">{exp_fp:.1f} Pts</span>
                                </div>
                            </div>
                            """)
                            
                            col = cols[idx % len(cols)]
                            col.markdown(card_html, unsafe_allow_html=True)
                            with col.expander("🧠 Why? (AI / SHAP)"):
                                if 'genai_explanations' in dream_team and player in dream_team['genai_explanations']:
                                    st.markdown(dream_team['genai_explanations'][player])
                                else:
                                    st.caption(exp.get('human_readable_explanation', ''))
                                    
                                    st.markdown("**FP Model Factors**")
                                    for f in exp.get('top_positive_fp_factors', [])[:2]:
                                        st.caption(f"🟢 `{f['feature']}`: +{f['contribution']:.1f}")
                                    for f in exp.get('top_negative_fp_factors', [])[:2]:
                                        st.caption(f"🔴 `{f['feature']}`: {f['contribution']:.1f}")
                                        
                                    st.markdown("**P(play) Factors**")
                                    for f in exp.get('top_positive_play_factors', [])[:2]:
                                        st.caption(f"🟢 `{f['feature']}`: +{f['contribution']:.2f}")
                                    for f in exp.get('top_negative_play_factors', [])[:2]:
                                        st.caption(f"🔴 `{f['feature']}`: {f['contribution']:.2f}")

                
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown("---")
                with st.expander("⏱️ System Latency Details"):
                    st.write(f"- **Inference Latency**: {latencies['total_inference']*1000:.1f} ms")
                    st.write(f"- **SHAP Latency**: {latencies.get('shap_explanation', 0)*1000:.1f} ms")
                    st.write(f"- **UI Overhead**: {latencies.get('ui_overhead', 0)*1000:.1f} ms")
                
            except ValueError as e:
                st.error(f"Prediction Error: {str(e)}")
                st.info("Ensure the selected teams have historically played a match near this date to form a candidate pool.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")
