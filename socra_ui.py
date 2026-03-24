"""
Socra – socra_ui.py
===================
AI Answer Verifier — let two AIs debate to find a better answer.

Run:
  cd socra
  streamlit run socra_ui.py
"""

import streamlit as st
import socra_core as core
import socra_db   as db

# ═══════════════════════════════════════════════════════════
# Page config  (must be first Streamlit call)
# ═══════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Socra – AI Answer Verifier",
    page_icon="✦",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════
# Translations
# ═══════════════════════════════════════════════════════════

T = {
    "en": {
        "subtitle":          "Not sure if AI's answer is right? Let another AI challenge it.",
        "lang_label":        "Language",
        "select_models":     "Select two models to debate",
        "not_configured":    "API key not set",
        "err_no_models":     "Please configure at least two API keys in your .env file.",
        "err_need_more":     "Please select one more model (need exactly 2).",
        "err_too_many":      "Please uncheck one model (need exactly 2, currently {n} selected).",
        "question_label":    "Your question",
        "question_ph":       "Ask anything — math, logic, facts… anything you want two AIs to cross-check",
        "image_label":       "📷 Upload image (optional)",
        "image_caption":     "Attach a photo of your question",
        "image_attached":    "📷 Image attached: {name}",
        "image_only_q":      "Please analyze and answer the question shown in the image.",
        "status_image":      "📷 Image included",
        "run_btn":           "▶ Verify",
        "hint_models":       "max 3 debate rounds",
        "status_running":    "🔍 Verifying...",
        "status_question":   "**Question:** ",
        "status_models":     "**Models:** ",
        "round0_done":       "⚡ **Round 0 complete** — both models answered independently",
        "confidence":        "confidence",
        "error_label":       "Error",
        "round_done":        "🔄 **Round {n} complete** — cross-examination",
        "agrees":            "✅ Agrees",
        "disagrees":         "❌ Disagrees",
        "converged_msg":     "✅ **Consensus reached at round {n}! Generating summary…**",
        "max_rounds_msg":    "⚠️ **Max rounds reached, still diverged — generating final summary…**",
        "status_done_ok":    "✅ Verified · {cost}",
        "status_done_fail":  "⚠️ Diverged · {cost}",
        "badge_verified":    "✅ Verified — consensus after {n} round(s)",
        "badge_disputed":    "⚠️ Diverged — models did not fully agree (summary below)",
        "answer_label_ok":   "✅ Verified Answer",
        "answer_label_fail": "⚠️ Best Answer (diverged)",
        "key_disagree_label":"⚔️ Key Disagreement",
        "agreement_score":   "Agreement score",
        "expand_debate":     "📖 Show full debate",
        "meta_cost":         "Total cost",
        "empty_title":       "Enter a question to start AI cross-verification",
        "empty_sub":         "Two AIs answer independently, then challenge each other until they agree",
        "round0_header":     "Round 0 — Independent Answers",
        "roundn_header":     "Round {n} — Cross-Examination",
        "refined_label":     "✏️ Refined answer:",
        "agrees_tag":        "✅ Agrees",
        "disagrees_tag":     "❌ Disagrees",
        "followup_label":    "💬 Ask a follow-up",
        "followup_ph":       "Not clear on something? Ask here…",
        "followup_btn":      "Ask",
        "followup_model":    "Answer with",
        "sidebar_title":     "📋 History",
        "sidebar_empty":     "No past questions yet.",
        "sidebar_clear":     "Clear all",
        "sidebar_verified":  "✅",
        "sidebar_disputed":  "⚠️",
        "new_question":      "+ New question",
    },
    "zh": {
        "subtitle":          "不确定 AI 的答案对不对？让另一个 AI 来挑战它。",
        "lang_label":        "语言",
        "select_models":     "选择两个模型进行辩论",
        "not_configured":    "未配置 API Key",
        "err_no_models":     "请在 .env 文件中配置至少两个 API Key。",
        "err_need_more":     "请再选一个模型（需要恰好 2 个）。",
        "err_too_many":      "已选 {n} 个，请取消其中一个（需要恰好 2 个）。",
        "question_label":    "你的问题",
        "question_ph":       "输入任何问题 — 数学题、逻辑题、事实问题……任何你想让两个 AI 交叉验证的内容",
        "image_label":       "📷 上传图片（可选）",
        "image_caption":     "上传题目截图或拍照",
        "image_attached":    "📷 已上传图片：{name}",
        "image_only_q":      "请分析并回答图片中的问题。",
        "status_image":      "📷 已附图片",
        "run_btn":           "▶ 开始验证",
        "hint_models":       "最多 3 轮辩论",
        "status_running":    "🔍 验证中...",
        "status_question":   "**问题：** ",
        "status_models":     "**模型：** ",
        "round0_done":       "⚡ **Round 0 完成** — 两个模型已独立作答",
        "confidence":        "信心度",
        "error_label":       "出错",
        "round_done":        "🔄 **Round {n} 完成** — 交叉审查",
        "agrees":            "✅ 同意",
        "disagrees":         "❌ 不同意",
        "converged_msg":     "✅ **第 {n} 轮达成共识！正在生成总结……**",
        "max_rounds_msg":    "⚠️ **已达最大轮次，仍存在分歧 — 正在生成综合结论……**",
        "status_done_ok":    "✅ 验证完成 · {cost}",
        "status_done_fail":  "⚠️ 存在分歧 · {cost}",
        "badge_verified":    "✅ 已验证 — {n} 轮后两模型达成共识",
        "badge_disputed":    "⚠️ 存在分歧 — 模型未完全一致（以下为综合结论）",
        "answer_label_ok":   "✅ 验证结论",
        "answer_label_fail": "⚠️ 综合结论（存在分歧）",
        "key_disagree_label":"⚔️ 核心分歧",
        "agreement_score":   "一致度",
        "expand_debate":     "📖 查看完整辩论过程",
        "meta_cost":         "总费用",
        "empty_title":       "输入问题，开始 AI 交叉验证",
        "empty_sub":         "两个 AI 独立回答，再互相审查，直到达成共识",
        "round0_header":     "Round 0 — 独立回答",
        "roundn_header":     "Round {n} — 交叉审查",
        "refined_label":     "✏️ 修正答案：",
        "agrees_tag":        "✅ 同意",
        "disagrees_tag":     "❌ 不同意",
        "followup_label":    "💬 追问",
        "followup_ph":       "有不明白的地方？继续问……",
        "followup_btn":      "提问",
        "followup_model":    "由谁回答",
        "sidebar_title":     "📋 历史记录",
        "sidebar_empty":     "暂无历史记录。",
        "sidebar_clear":     "清空全部",
        "sidebar_verified":  "✅",
        "sidebar_disputed":  "⚠️",
        "new_question":      "+ 新问题",
    },
}

# ═══════════════════════════════════════════════════════════
# Custom CSS
# ═══════════════════════════════════════════════════════════

st.markdown("""
<style>
.block-container { max-width: 740px; padding-top: 2.2rem; padding-bottom: 3rem; }
.socra-title {
    font-size: 2.4rem; font-weight: 900; letter-spacing: -1.5px;
    color: #0f172a; margin-bottom: 0; line-height: 1;
}
.socra-accent { color: #6366f1; }
.socra-sub { font-size: 0.95rem; color: #64748b; margin-top: 6px; margin-bottom: 1.2rem; }
.sec-label {
    font-size: 0.78rem; font-weight: 700; color: #94a3b8;
    text-transform: uppercase; letter-spacing: 0.07em; margin-bottom: 8px;
}
.badge-verified {
    display: inline-flex; align-items: center; gap: 6px;
    background: #dcfce7; color: #15803d;
    padding: 5px 16px; border-radius: 20px;
    font-weight: 700; font-size: 0.88rem; margin-bottom: 14px;
}
.badge-disputed {
    display: inline-flex; align-items: center; gap: 6px;
    background: #fef9c3; color: #92400e;
    padding: 5px 16px; border-radius: 20px;
    font-weight: 700; font-size: 0.88rem; margin-bottom: 14px;
}
.answer-box-verified {
    background: #f0fdf4; border: 2px solid #86efac;
    border-radius: 14px; padding: 1.3rem 1.6rem; margin: 0.4rem 0 0.8rem 0;
}
.answer-box-disputed {
    background: #fffbeb; border: 2px solid #fcd34d;
    border-radius: 14px; padding: 1.3rem 1.6rem; margin: 0.4rem 0 0.8rem 0;
}
.answer-label {
    font-size: 0.72rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.08em; color: #6366f1; margin-bottom: 8px;
}
.answer-text { font-size: 1.05rem; color: #0f172a; line-height: 1.75; }
.summary-strip {
    background: #f8fafc; border-left: 3px solid #6366f1;
    padding: 0.75rem 1rem; border-radius: 0 8px 8px 0;
    font-size: 0.88rem; color: #475569; margin-bottom: 1rem;
}
.disagree-box {
    background: #fff1f2; border: 1.5px solid #fecaca;
    border-radius: 10px; padding: 0.7rem 1rem; margin-bottom: 0.8rem;
    font-size: 0.88rem; color: #991b1b;
}
.disagree-label {
    font-size: 0.7rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.07em; color: #dc2626; margin-bottom: 4px;
}
.score-bar-wrap {
    background: #f1f5f9; border-radius: 8px; height: 8px;
    width: 100%; margin: 4px 0 12px 0; overflow: hidden;
}
.score-bar-fill {
    height: 8px; border-radius: 8px;
    background: linear-gradient(90deg, #ef4444, #f59e0b, #22c55e);
}
.followup-box {
    background: #f8fafc; border: 1.5px solid #e2e8f0;
    border-radius: 14px; padding: 1rem 1.2rem; margin-top: 1.2rem;
}
.followup-answer {
    background: #eff6ff; border-left: 3px solid #6366f1;
    border-radius: 0 8px 8px 0; padding: 0.75rem 1rem;
    font-size: 0.92rem; color: #1e293b; line-height: 1.7;
    margin-top: 0.5rem; margin-bottom: 0.8rem;
}
.round-divider {
    text-align: center; font-size: 0.72rem; font-weight: 600;
    color: #cbd5e1; margin: 16px 0 10px 0; letter-spacing: 0.06em;
    text-transform: uppercase;
}
.bubble-left {
    background: #fff7ed; border: 1px solid #fed7aa;
    border-radius: 14px 14px 14px 3px;
    padding: 10px 15px; margin: 5px 0; margin-right: 18%;
    font-size: 0.87rem; color: #1e293b; line-height: 1.55;
}
.bubble-right {
    background: #eff6ff; border: 1px solid #bfdbfe;
    border-radius: 14px 14px 3px 14px;
    padding: 10px 15px; margin: 5px 0; margin-left: 18%;
    font-size: 0.87rem; color: #1e293b; line-height: 1.55;
}
.bubble-name  { font-size: 0.72rem; font-weight: 700; color: #94a3b8; margin-bottom: 4px; }
.tag-agree    { color: #16a34a; font-weight: 700; font-size: 0.82rem; }
.tag-disagree { color: #dc2626; font-weight: 700; font-size: 0.82rem; }
.tag-conf     { color: #6366f1; font-size: 0.78rem; }
.tag-score    { color: #f59e0b; font-size: 0.78rem; }
.refined-note {
    color: #6366f1; font-size: 0.8rem; margin-top: 5px;
    border-top: 1px solid #e0e7ff; padding-top: 5px;
}
.key-dis-inline {
    color: #dc2626; font-size: 0.78rem; margin-top: 4px;
    border-top: 1px solid #fecaca; padding-top: 4px;
}
.empty-state  { text-align: center; padding: 3.5rem 1rem; color: #94a3b8; }
.empty-icon   { font-size: 2.8rem; margin-bottom: 0.8rem; }
.empty-title  { font-size: 1rem; font-weight: 600; color: #64748b; margin-bottom: 4px; }
.empty-sub    { font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# Session state
# ═══════════════════════════════════════════════════════════

defaults = {
    "result":        None,
    "lang":          "zh",
    "followups":     [],
    "active_q":      "",
    "active_models": [],
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ═══════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════

def t(key: str, **kwargs) -> str:
    s = T[st.session_state.lang].get(key, key)
    return s.format(**kwargs) if kwargs else s

MODEL_META = {
    "Claude": {"icon": "🟠"},
    "GPT-4":  {"icon": "🟢"},
    "Gemini": {"icon": "🔵"},
}

def icon(model: str) -> str:
    return MODEL_META.get(model, {}).get("icon", "⚪")

def bubble_class(model: str, models: list) -> str:
    return "bubble-left" if models.index(model) == 0 else "bubble-right"

def render_debate_log(debate_log: list, model_a: str, model_b: str):
    models = [model_a, model_b]
    for entry in debate_log:
        rnd    = entry["round"]
        header = t("round0_header") if rnd == 0 else t("roundn_header", n=rnd)
        st.markdown(
            f'<div class="round-divider">─── {header} ───</div>',
            unsafe_allow_html=True,
        )
        for model, res in entry["responses"].items():
            bc = bubble_class(model, models) if model in models else "bubble-left"
            if not res.get("ok"):
                st.markdown(
                    f'<div class="{bc}"><span class="bubble-name">{icon(model)} {model}</span> '
                    f'<span style="color:#ef4444">{t("error_label")}: {res.get("error","")}</span></div>',
                    unsafe_allow_html=True,
                )
                continue
            d = res["data"]
            if rnd == 0:
                conf      = d.get("confidence", 0)
                answer    = (d.get("answer", "") or "").replace("\n", "<br>")
                reasoning = (d.get("reasoning", "") or "")[:220]
                content   = (
                    f'<div class="bubble-name">{icon(model)} {model} '
                    f'<span class="tag-conf">· {t("confidence")} {conf}%</span></div>'
                    f'{answer}'
                )
                if reasoning:
                    ellipsis = "…" if len(d.get("reasoning", "")) > 220 else ""
                    content += f'<div style="color:#94a3b8;font-size:0.8rem;margin-top:5px">💭 {reasoning}{ellipsis}</div>'
            else:
                agrees   = d.get("agrees", False)
                score    = d.get("agreement_score")
                key_dis  = d.get("key_disagreement") or ""
                critique = (d.get("critique", "") or "").replace("\n", "<br>")
                refined  = d.get("refined_answer", "") or ""
                atag     = f'<span class="tag-agree">{t("agrees_tag")}</span>' if agrees else f'<span class="tag-disagree">{t("disagrees_tag")}</span>'
                stag     = f'<span class="tag-score">· {t("agreement_score")} {score}%</span>' if score is not None else ""
                content  = f'<div class="bubble-name">{icon(model)} {model} {atag} {stag}</div>{critique}'
                if key_dis and not agrees:
                    content += f'<div class="key-dis-inline">⚔️ {key_dis}</div>'
                if refined and not agrees:
                    short    = refined[:200] + ("…" if len(refined) > 200 else "")
                    content += f'<div class="refined-note">{t("refined_label")} {short}</div>'
            st.markdown(f'<div class="{bc}">{content}</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# Sidebar — History
# ═══════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(f"### {t('sidebar_title')}")

    if st.button(t("new_question"), use_container_width=True):
        st.session_state.result        = None
        st.session_state.followups     = []
        st.session_state.active_q      = ""
        st.session_state.active_models = []
        st.rerun()

    history = db.get_history(30)
    if not history:
        st.caption(t("sidebar_empty"))
    else:
        for row in history:
            badge   = t("sidebar_verified") if row["converged"] else t("sidebar_disputed")
            q_short = row["question"][:55] + ("…" if len(row["question"]) > 55 else "")
            label   = f"{badge} {q_short}"
            if st.button(label, key=f"hist_{row['id']}", help=row["created_at"][:10], use_container_width=True):
                session = db.get_session(row["id"])
                if session and session.get("result"):
                    st.session_state.result        = session["result"]
                    st.session_state.active_q      = session["question"]
                    st.session_state.active_models = [session["model_a"], session["model_b"]]
                    st.session_state.followups     = []
                    st.session_state.lang          = session.get("lang", "zh")
                    st.rerun()

        st.divider()
        if st.button(t("sidebar_clear"), use_container_width=True):
            db.clear_history()
            st.rerun()

# ═══════════════════════════════════════════════════════════
# Header
# ═══════════════════════════════════════════════════════════

st.markdown(
    '<p class="socra-title">✦ <span class="socra-accent">Socra</span></p>',
    unsafe_allow_html=True,
)

lang_choice = st.radio(
    t("lang_label"),
    options=["中文", "English"],
    horizontal=True,
    index=0 if st.session_state.lang == "zh" else 1,
    label_visibility="collapsed",
)
st.session_state.lang = "zh" if lang_choice == "中文" else "en"

st.markdown(
    f'<p class="socra-sub">{t("subtitle")}</p>',
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════════════
# Model selector
# ═══════════════════════════════════════════════════════════

api_status = core.get_api_status()
st.markdown(f'<p class="sec-label">{t("select_models")}</p>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
selected_models = []

for col, name in zip([c1, c2, c3], ["Claude", "GPT-4", "Gemini"]):
    avail = api_status[name]
    with col:
        checked = st.checkbox(
            f"{icon(name)} {name}",
            value=avail,
            disabled=not avail,
            key=f"sel_{name}",
        )
        if not avail:
            st.caption(t("not_configured"))
        if checked and avail:
            selected_models.append(name)

if len(selected_models) == 0:
    st.error(t("err_no_models"))
elif len(selected_models) == 1:
    st.warning(t("err_need_more"))
elif len(selected_models) > 2:
    st.warning(t("err_too_many", n=len(selected_models)))

# ═══════════════════════════════════════════════════════════
# Question input + image upload
# ═══════════════════════════════════════════════════════════

st.markdown("<br>", unsafe_allow_html=True)
st.markdown(f'<p class="sec-label">{t("question_label")}</p>', unsafe_allow_html=True)
question = st.text_area(
    label="q",
    label_visibility="collapsed",
    placeholder=t("question_ph"),
    height=110,
    key="question_input",
)

uploaded_img = st.file_uploader(
    t("image_label"),
    type=["png", "jpg", "jpeg", "gif", "webp"],
    help=t("image_caption"),
    key="img_upload",
)

import base64 as _b64
image_b64  = None
image_mime = "image/jpeg"
if uploaded_img is not None:
    image_bytes = uploaded_img.read()
    image_b64   = _b64.b64encode(image_bytes).decode()
    image_mime  = uploaded_img.type or "image/jpeg"
    st.caption(t("image_attached", name=uploaded_img.name))

has_input = bool(question.strip()) or (image_b64 is not None)

col_btn, col_hint = st.columns([1.3, 3])
with col_btn:
    run_btn = st.button(
        t("run_btn"),
        type="primary",
        disabled=(len(selected_models) != 2 or not has_input),
        use_container_width=True,
    )
with col_hint:
    if len(selected_models) == 2:
        ma, mb = selected_models
        st.caption(f"{icon(ma)} {ma} vs {icon(mb)} {mb} · {t('hint_models')}")

# ═══════════════════════════════════════════════════════════
# Run debate
# ═══════════════════════════════════════════════════════════

if run_btn and has_input and len(selected_models) == 2:
    st.session_state.result    = None
    st.session_state.followups = []
    model_a, model_b = selected_models[0], selected_models[1]
    run_lang    = st.session_state.lang
    effective_q = question.strip() or t("image_only_q")

    with st.status(t("status_running"), expanded=True) as status:
        q_display = question.strip() or f"[{t('status_image')}]"
        st.write(f"{t('status_question')}{q_display[:100]}{'…' if len(q_display) > 100 else ''}")
        if image_b64:
            st.write(t("status_image"))
        st.write(f"{t('status_models')}{icon(model_a)} {model_a}  vs  {icon(model_b)} {model_b}")
        st.divider()

        def on_update(event: str, data: dict):
            if event == "round_complete":
                rnd       = data["round"]
                responses = data["responses"]
                if rnd == 0:
                    st.write(T[run_lang]["round0_done"])
                    for model, res in responses.items():
                        if res.get("ok"):
                            conf     = res["data"].get("confidence", 0)
                            preview  = (res["data"].get("answer") or "")[:100]
                            ellipsis = "…" if len(res["data"].get("answer", "")) > 100 else ""
                            st.write(f"  {icon(model)} **{model}** ({T[run_lang]['confidence']} {conf}%): {preview}{ellipsis}")
                        else:
                            st.write(f"  ❌ {icon(model)} **{model}** {T[run_lang]['error_label']}: {res.get('error','')}")
                else:
                    st.write(T[run_lang]["round_done"].format(n=rnd))
                    for model, res in responses.items():
                        if res.get("ok"):
                            agrees = res["data"].get("agrees", False)
                            score  = res["data"].get("agreement_score", "?")
                            mark   = T[run_lang]["agrees"] if agrees else T[run_lang]["disagrees"]
                            st.write(f"  {icon(model)} **{model}**: {mark} · score {score}%")
            elif event == "converged":
                st.write(T[run_lang]["converged_msg"].format(n=data["round"]))
            elif event == "max_rounds":
                st.write(T[run_lang]["max_rounds_msg"])

        result = core.run_debate(
            effective_q, model_a, model_b,
            max_rounds=3, lang=run_lang, on_update=on_update,
            image_b64=image_b64, image_mime=image_mime,
        )

        cost_str = f"${result.get('total_cost', 0):.4f}"
        if result.get("converged"):
            status.update(label=T[run_lang]["status_done_ok"].format(cost=cost_str), state="complete", expanded=False)
        else:
            status.update(label=T[run_lang]["status_done_fail"].format(cost=cost_str), state="complete", expanded=False)

    db.save_session(effective_q, result, lang=run_lang)
    st.session_state.result        = result
    st.session_state.active_q      = effective_q
    st.session_state.active_models = [model_a, model_b]

# ═══════════════════════════════════════════════════════════
# Display results
# ═══════════════════════════════════════════════════════════

result = st.session_state.result

if result:
    converged = result.get("converged", False)
    model_a   = result.get("model_a", "")
    model_b   = result.get("model_b", "")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Status badge ──────────────────────────────────────
    if converged:
        rnd = result.get("converged_at_round", 1)
        st.markdown(
            f'<div class="badge-verified">{t("badge_verified", n=rnd)}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="badge-disputed">{t("badge_disputed")}</div>',
            unsafe_allow_html=True,
        )

    # ── Key disagreement highlight ────────────────────────
    key_dis = result.get("key_disagreement", "") or ""
    if key_dis and not converged:
        st.markdown(
            f'<div class="disagree-box">'
            f'<div class="disagree-label">{t("key_disagree_label")}</div>'
            f'{key_dis}'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Agreement score bar ───────────────────────────────
    last_round = next(
        (e for e in reversed(result.get("debate_log", [])) if e["round"] > 0),
        None
    )
    if last_round:
        scores = [
            r["data"].get("agreement_score", 0)
            for r in last_round["responses"].values()
            if r.get("ok") and r["data"].get("agreement_score") is not None
        ]
        if scores:
            avg_score = int(sum(scores) / len(scores))
            st.markdown(
                f'<div style="font-size:0.78rem;color:#94a3b8;margin-bottom:2px">'
                f'{t("agreement_score")}: <b>{avg_score}%</b></div>'
                f'<div class="score-bar-wrap">'
                f'<div class="score-bar-fill" style="width:{avg_score}%"></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── Final answer ──────────────────────────────────────
    box_class  = "answer-box-verified" if converged else "answer-box-disputed"
    box_label  = t("answer_label_ok") if converged else t("answer_label_fail")
    final_html = (result.get("final_answer") or "").replace("\n", "<br>")
    st.markdown(
        f'<div class="{box_class}">'
        f'<div class="answer-label">{box_label}</div>'
        f'<div class="answer-text">{final_html}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Verification summary ──────────────────────────────
    summary = result.get("verification_summary", "")
    if summary:
        st.markdown(
            f'<div class="summary-strip">📋 {summary}</div>',
            unsafe_allow_html=True,
        )

    # ── Full debate log ───────────────────────────────────
    with st.expander(t("expand_debate"), expanded=False):
        render_debate_log(result.get("debate_log", []), model_a, model_b)

    # ── Meta bar ──────────────────────────────────────────
    cost = result.get("total_cost", 0)
    st.markdown(
        f'<p style="font-size:0.76rem;color:#cbd5e1;margin-top:0.5rem;">'
        f'💰 {t("meta_cost")} ${cost:.4f} · '
        f'{icon(model_a)} {model_a} vs {icon(model_b)} {model_b}'
        f'</p>',
        unsafe_allow_html=True,
    )

    # ── Follow-up conversation ────────────────────────────
    # Fix 1: use st.container(border=True) instead of raw HTML div —
    #         HTML divs don't wrap Streamlit widgets in practice.
    with st.container(border=True):
        st.markdown(f'<p class="sec-label">{t("followup_label")}</p>', unsafe_allow_html=True)

        for fu in st.session_state.followups:
            answered_by = fu.get("model", "")
            st.markdown(f"**{fu['question']}**")
            st.markdown(
                f'<div class="followup-answer">'
                f'{fu["answer"].replace(chr(10), "<br>")}'
                f'<div style="font-size:0.72rem;color:#94a3b8;margin-top:6px">'
                f'{icon(answered_by)} {answered_by}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Model selector for follow-up
        available_models = st.session_state.active_models or [model_a, model_b]
        fu_model_options = [f"{icon(m)} {m}" for m in available_models]
        fu_model_choice  = st.radio(
            t("followup_model"),
            options=fu_model_options,
            horizontal=True,
            key="fu_model_radio",
        )
        fu_model = available_models[fu_model_options.index(fu_model_choice)]

        fu_col1, fu_col2 = st.columns([4, 1])
        with fu_col1:
            fu_input = st.text_input(
                "followup",
                label_visibility="collapsed",
                placeholder=t("followup_ph"),
                key="fu_input",
            )
        with fu_col2:
            fu_btn = st.button(
                t("followup_btn"),
                disabled=not fu_input.strip(),
                use_container_width=True,
            )

        if fu_btn and fu_input.strip():
            with st.spinner("…"):
                fu_result = core.run_followup(
                    original_question = st.session_state.active_q,
                    debate_result     = result,
                    followup_question = fu_input.strip(),
                    model             = fu_model,
                    lang              = st.session_state.lang,
                )
            if fu_result["ok"]:
                st.session_state.followups.append({
                    "question": fu_input.strip(),
                    "answer":   fu_result["answer"],
                    "cost":     fu_result["cost"],
                    "model":    fu_model,
                })
            else:
                st.error(f"❌ {fu_result['answer']}")
            st.rerun()

elif not run_btn:
    st.markdown(
        f'<div class="empty-state">'
        f'<div class="empty-icon">✦</div>'
        f'<div class="empty-title">{t("empty_title")}</div>'
        f'<div class="empty-sub">{t("empty_sub")}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
