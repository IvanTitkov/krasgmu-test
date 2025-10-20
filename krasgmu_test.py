import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime, os

st.set_page_config(page_title="Я — будущий врач?! (v7)", layout="centered", initial_sidebar_state="collapsed")

try:
    st.image("Ресурс 1 (2).png", width=140)
except Exception:
    pass

st.markdown("# Методика оценки готовности к медицинскому образованию")
st.caption("Разработано кафедрой клинической психологии и педагогики КрасГМУ, Красноярск, 2025")

# --- CSS ---
st.markdown("""
<style>
.question-text { font-size: 18px !important; line-height: 1.6; font-weight: 500; margin: 10px 0 0 0; }
.question-slider { margin-top: -6px; margin-bottom: 12px; }
@media (max-width: 600px) {
  .question-text {font-size: 16px !important;}
  div[data-testid="stSlider"] label {font-size: 14px !important;}
}
</style>
""", unsafe_allow_html=True)

# --- Константы ---
GATES_VO = {"STAB":55, "BIO_TOL":55}
GATES_SPO = {"STAB":50, "BIO_TOL":50}
GATE_PENALTY = 0.85
IGMO_WEIGHTS = {
    "HELP": 0.15, "EMPA": 0.15, "PUBH": 0.10,
    "STAB": 0.15, "BIO_TOL": 0.15,
    "PREC": 0.10, "SCI": 0.10, "TECH": 0.10
}
IGMO_EXTRA_K = 0.8
IGMO_QC_MAX_PENALTY = 5.0

SCALE_LABELS = {
    'HELP': 'Стремление помогать',
    'EMPA': 'Эмпатия',
    'STAB': 'Устойчивость',
    'PREC': 'Внимательность',
    'MANUAL': 'Ловкость и координация движений',
    'BIO_TOL': 'Спокойное отношение к мед. процедурам',
    'TECH': 'Интерес к медицинским технологиям',
    'SCI': 'Исследование',
    'PEDI': 'Работа с детьми',
    'PUBH': 'Интерес к профилактике',
}

SCALES = {
    'HELP': [
        {"text":"Мне важно, чтобы моя работа приносила пользу людям.", "rev":False},
        {"text":"Мне приятно видеть, что благодаря мне человеку становится лучше.", "rev":False},
        {"text":"Я готов помогать даже когда это требует усилий.", "rev":False},
        {"text":"Я ощущаю смысл в том, чтобы посвятить жизнь помощи людям и развитию медицины.", "rev":False, "extra_gmo": True},
    ],
    'EMPA': [
        {"text":"Я легко понимаю, что чувствует другой человек.", "rev":False},
        {"text":"В трудном разговоре я сохраняю уважительный тон.", "rev":False},
        {"text":"Чужие эмоции часто меня раздражают.", "rev":True},
    ],
    'STAB': [
        {"text":"В стрессовой ситуации я быстро беру себя в руки.", "rev":False},
        {"text":"Под давлением ответственности я не теряюсь.", "rev":False},
        {"text":"Даже когда вокруг напряжённо, я сохраняю работоспособность.", "rev":False},
    ],
    'PREC': [
        {"text":"Я стараюсь делать всё точно и аккуратно.", "rev":False},
        {"text":"Быстро замечаю ошибки в записях и цифрах.", "rev":False},
        {"text":"Подробные правила и инструкции меня быстро утомляют.", "rev":True},
    ],
    'MANUAL': [
        {"text":"Мне нравится кропотливая работа руками.", "rev":False},
        {"text":"У меня хорошая координация мелких движений.", "rev":False},
        {"text":"Я терпеливо довожу ручную работу до идеала.", "rev":False},
    ],
    'BIO_TOL': [
        {"text":"Вид крови и перевязок не выводит меня из равновесия.", "rev":False},
        {"text":"Я спокойно отношусь к больницам и операциям.", "rev":False},
        {"text":"Виды медицинских процедур меня пугают.", "rev":True},
    ],
    'TECH': [
        {"text":"Мне интересно разбираться в приборах и программах.", "rev":False},
        {"text":"Люблю анализировать данные и находить закономерности.", "rev":False},
        {"text":"Технические детали мне обычно неинтересны.", "rev":True},
    ],
    'SCI': [
        {"text":"Мне нравится проверять гипотезы и делать выводы.", "rev":False},
        {"text":"Мне интересны опыты и эксперименты.", "rev":False},
        {"text":"У меня есть терпение собирать и сравнивать факты.", "rev":False},
        {"text":"Мне интересно узнавать, как устроено тело человека и как работают лекарства.", "rev":False, "extra_gmo": True},
    ],
    'PEDI': [
        {"text":"Мне комфортно общаться с детьми.", "rev":False},
        {"text":"Терпело объясняю одно и то же ребёнку и его родителям.", "rev":False},
        {"text":"Мне нравится заботиться о младших.", "rev":False},
    ],
    'PUBH': [
        {"text":"Мне важно, чтобы соблюдались правила чистоты и безопасности.", "rev":False},
        {"text":"Меня интересуют вопросы здоровья всего сообщества.", "rev":False},
        {"text":"Мне близка идея предотвращать болезни, а не только лечить.", "rev":False},
    ],
}

ORDER = list(SCALES.keys())

def reverse_score(v): return 6 - v

def compute_profile(answers):
    prof = {}
    for scale, items in SCALES.items():
        vals = []
        for idx, item in enumerate(items):
            v = answers.get(f"{scale}_{idx}", 3)
            if item.get("rev", False): v = reverse_score(v)
            vals.append(v)
        prof[scale] = (np.mean(vals) - 1) / 4 * 100.0
    return prof

def apply_level_gates(level, profile):
    gates = GATES_VO if level.startswith("Высшего") else GATES_SPO
    return [(sc, thr, profile.get(sc, 0.0)) for sc, thr in gates.items() if profile.get(sc, 0.0) < thr]

def compute_igmo(profile, answers, level):
    base = sum(IGMO_WEIGHTS[sc] * profile.get(sc, 0.0) for sc in IGMO_WEIGHTS)
    fails = apply_level_gates(level, profile)
    if fails: base *= (GATE_PENALTY ** len(fails))
    delta = IGMO_EXTRA_K * ((answers.get("HELP_3", 3) - 3) + (answers.get("SCI_3", 3) - 3))
    igmo = np.clip(base + delta, 0.0, 100.0)
    qc_flags = []
    sdi_keys = [("HELP",0),("EMPA",1),("PUBH",2)]
    sdi_vals = [reverse_score(answers[f"{s}_{i}"]) if SCALES[s][i]["rev"] else answers[f"{s}_{i}"] for s,i in sdi_keys]
    sdi = float(np.mean(sdi_vals)) if sdi_vals else 3.0
    if sdi >= 4.6: qc_flags.append("Социальная желательность (высокая)")
    pairs = [("HELP",0,"HELP",2),("PREC",0,"PREC",1)]
    for a,b,c,d in pairs:
        av,bv = answers[f"{a}_{b}"],answers[f"{c}_{d}"]
        if SCALES[a][b].get("rev"): av=reverse_score(av)
        if SCALES[c][d].get("rev"): bv=reverse_score(bv)
        if abs(av-bv)>2: qc_flags.append("Низкая согласованность ответов"); break
    vals=[]
    for sc,itms in SCALES.items():
        for i,it in enumerate(itms):
            v=answers[f"{sc}_{i}"]
            if it.get("rev"): v=reverse_score(v)
            vals.append(v)
    if np.std(vals)<0.5: qc_flags.append("Однообразные ответы")
    qc_penalty=min(IGMO_QC_MAX_PENALTY,3*len(qc_flags))
    return max(0.0, igmo-qc_penalty), fails, qc_flags, sdi

def save_to_csv(answers, profile, igmo, sdi, qc_flags, level):
    data = {"timestamp": datetime.datetime.now().isoformat(),
            "level": level, "IGMO": round(igmo,2),
            "SDI": round(sdi,2),
            "QC_flags": "; ".join(qc_flags) if qc_flags else ""}
    for sc,val in profile.items(): data[sc]=round(val,2)
    df=pd.DataFrame([data])
    fn="results_krasgmu.csv"
    df.to_csv(fn, mode="a", header=not os.path.exists(fn),
              index=False, encoding="utf-8-sig")

st.markdown("### Уровень планируемого образования")
level=st.radio("Ты планируешь изучать медицину на уровне…",
               ["Высшего образования (ВО)", "Среднего профессионального образования (СПО)"])
st.divider()

with st.form("short_form"):
    answers={}
    n=1
    for sc in ORDER:
        for i,it in enumerate(SCALES[sc]):
            st.markdown(f"<p class='question-text'>{n}. {it['text']}</p>", unsafe_allow_html=True)
            answers[f"{sc}_{i}"]=st.slider("",1,5,3,step=1,key=f"{sc}_{i}")
            n+=1
    submitted=st.form_submit_button("Готово — показать результат")

def radar_chart(profile):
    labels = list(ORDER)
    values = [profile[k] for k in labels] + [profile[labels[0]]]
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist() + [0]

    fig = plt.figure(figsize=(6,6))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, values, color="tab:blue")
    ax.fill(angles, values, alpha=.15, color="tab:blue")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([SCALE_LABELS[k] for k in labels], fontsize=10)  # <- без labelpad
    ax.tick_params(axis='x', pad=12)  # <- отступ подписей от оси

    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_ylim(0, 100)
    ax.tick_params(pad=8)
    return fig


if submitted:
    profile=compute_profile(answers)
    igmo,fails,qc_flags,sdi=compute_igmo(profile,answers,level)
    st.markdown("## Индекс готовности к медицинскому образованию (ИГМО)")
    msg=f"ИГМО: **{igmo:.1f} / 100**"
    if igmo>=80: st.success(msg+" — высокий уровень.")
    elif igmo>=60: st.info(msg+" — умеренный уровень.")
    elif igmo>=40: st.warning(msg+" — пограничный уровень.")
    else: st.error(msg+" — низкий уровень.")
    st.caption("ИГМО отражает общий уровень личностно-психологической подготовки к обучению в медицинском вузе.")

    if fails:
        ftxt="; ".join([f"{SCALE_LABELS[s]} < {t} (у вас {profile[s]:.0f})" for s,t,_ in fails])
        st.warning("Критические пороги не достигнуты — "+ftxt)
    if qc_flags:
        st.caption(f"Флаги качества: {', '.join(qc_flags)}  (SDI={sdi:.2f})")

    dfp=pd.DataFrame({"Шкала":[SCALE_LABELS[k] for k in ORDER],
                      "Баллы":[round(profile[k],1) for k in ORDER]})
    st.dataframe(dfp, use_container_width=True)
    st.pyplot(radar_chart(profile))

    save_to_csv(answers, profile, igmo, sdi, qc_flags, level)

    # --- Подбор подходящих направлений (калибровано по центроидам) ---
st.markdown("## Подходящие медицинские специальности")

h = profile

if level.startswith("Высшего"):

    # ЛЕЧЕБНОЕ ДЕЛО — ядро: STAB/BIO_TOL/PREC + HELP
    if (h['STAB'] >= 70 and h['BIO_TOL'] >= 70 and h['PREC'] >= 60 and h['HELP'] >= 70):
        st.markdown("**Лечебное дело** — высокий уровень устойчивости, толерантности к процедурам и внимательности при выраженной гуманистической мотивации.")

    # ПЕДИАТРИЯ — как лечебное дело + усиленные EMPA и PEDI
    if (h['STAB'] >= 70 and h['BIO_TOL'] >= 65 and h['PREC'] >= 60 and h['HELP'] >= 75
        and h['EMPA'] >= 80 and h['PEDI'] >= 75):
        st.markdown("**Педиатрия** — к профилю лечебного дела добавляются высокая эмпатия и выраженная склонность к работе с детьми.")

    # СТОМАТОЛОГИЯ — высокий MANUAL/BIO_TOL/STAB + PREC
    if (h['MANUAL'] >= 80 and h['BIO_TOL'] >= 80 and h['STAB'] >= 75 and h['PREC'] >= 75):
        st.markdown("**Стоматология** — выражены мануальные навыки, стрессоустойчивость, толерантность к процедурам и аккуратность.")

    # МЕДИКО-ПРОФИЛАКТИЧЕСКОЕ ДЕЛО — высокий PUBH/TECH + PREC
    if (h['PUBH'] >= 75 and h['TECH'] >= 65 and h['PREC'] >= 75):
        st.markdown("**Медико-профилактическое дело** — интерес к здоровью общества и технологиям при высокой организованности и внимании к регламентам.")

    # КЛИНИЧЕСКАЯ ПСИХОЛОГИЯ — высокий HELP/EMPA, BIO_TOL умеренный
    if (h['HELP'] >= 75 and h['EMPA'] >= 80):
        st.markdown("**Клиническая психология** — выраженная гуманистическая направленность и эмпатия.")

    # МЕДИЦИНСКАЯ КИБЕРНЕТИКА — высокий TECH/SCI
    if (h['TECH'] >= 80 and h['SCI'] >= 80):
        st.markdown("**Медицинская кибернетика** — сильная ориентация на технологии и исследовательскую деятельность.")

    # МЕДИЦИНСКАЯ БИОФИЗИКА — TECH высокий, SCI очень высокий
    if (h['TECH'] >= 75 and h['SCI'] >= 85):
        st.markdown("**Медицинская биофизика** — исследовательская направленность и интерес к устройству биологических систем.")

else:
    # СПО

    # СЕСТРИНСКОЕ ДЕЛО — HELP/EMPA + STAB/BIO_TOL + PREC
    if (h['HELP'] >= 70 and h['EMPA'] >= 70 and h['STAB'] >= 70 and h['BIO_TOL'] >= 65 and h['PREC'] >= 65):
        st.markdown("**Сестринское дело (СПО)** — сочетаются гуманистическая направленность, эмпатия и устойчивость к мед. среде.")

    # ЛАБОРАТОРНАЯ ДИАГНОСТИКА — PREC/TECH + SCI + BIO_TOL
    if (h['PREC'] >= 80 and h['TECH'] >= 70 and h['SCI'] >= 60 and h['BIO_TOL'] >= 60):
        st.markdown("**Лабораторная диагностика (СПО)** — выражены точность, технологичность и исследовательская мотивация.")

    # ФАРМАЦИЯ — высокий PREC + TECH (контакт умеренный)
    if (h['PREC'] >= 80 and h['TECH'] >= 70):
        st.markdown("**Фармация (СПО)** — внимание к деталям, регламентам и технологиям лекарств.")

    # МЕДИЦИНСКИЙ МАССАЖ — высокий MANUAL + EMPA + STAB
    if (h['MANUAL'] >= 80 and h['EMPA'] >= 65 and h['STAB'] >= 65):
        st.markdown("**Медицинский массаж (СПО)** — развитые мануальные навыки, эмпатия и выдержка.")


    st.markdown('---')
    st.caption("Схема ИГМО: взвешенные 8 базовых шкал + пороги по STAB/BIO_TOL (уровневые) + микрокоррекция двумя «ядерными» пунктами + поправка за качество ответов.")
