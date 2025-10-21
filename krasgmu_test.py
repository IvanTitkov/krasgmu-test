import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime, os

# ----------------------------------------------------------
# НАСТРОЙКИ СТРАНИЦЫ
# ----------------------------------------------------------
st.set_page_config(page_title="ИГМО", layout="centered", initial_sidebar_state="collapsed")

try:
    st.image("Ресурс 1 (2).png", width=140)
except Exception:
    pass

st.markdown("# Методика оценки готовности к медицинскому образованию")
st.caption("Разработано кафедрой клинической психологии и педагогики КрасГМУ, Красноярск, 2025")

# ----------------------------------------------------------
# CSS
# ----------------------------------------------------------
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

# ----------------------------------------------------------
# КОНСТАНТЫ
# ----------------------------------------------------------
GATES_VO = {"STAB":58, "BIO_TOL":58}
GATES_SPO = {"STAB":55, "BIO_TOL":55}
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
    'BIO_TOL': 'Толерантность к мед. процедурам',
    'TECH': 'Интерес к технологиям',
    'SCI': 'Исследовательский интерес',
    'PEDI': 'Работа с детьми',
    'PUBH': 'Интерес к профилактике',
}

# ----------------------------------------------------------
# ЭТАЛОННЫЕ ЦЕНТРОИДЫ
# ----------------------------------------------------------
SCALE_ORDER = ['HELP','EMPA','STAB','PREC','MANUAL','BIO_TOL','TECH','SCI','PEDI','PUBH']

CENTROIDS_VO = {
    'Лечебное дело':                 {'HELP':80,'EMPA':75,'STAB':80,'PREC':65,'MANUAL':55,'BIO_TOL':80,'TECH':55,'SCI':60,'PEDI':30,'PUBH':45},
    'Педиатрия':                     {'HELP':85,'EMPA':90,'STAB':75,'PREC':60,'MANUAL':45,'BIO_TOL':70,'TECH':45,'SCI':55,'PEDI':90,'PUBH':40},
    'Стоматология':                  {'HELP':70,'EMPA':70,'STAB':80,'PREC':80,'MANUAL':90,'BIO_TOL':85,'TECH':55,'SCI':50,'PEDI':20,'PUBH':30},
    'Медико-профилактическое дело':  {'HELP':70,'EMPA':60,'STAB':75,'PREC':85,'MANUAL':40,'BIO_TOL':65,'TECH':70,'SCI':65,'PEDI':25,'PUBH':90},
    'Клиническая психология':        {'HELP':85,'EMPA':90,'STAB':70,'PREC':60,'MANUAL':30,'BIO_TOL':40,'TECH':40,'SCI':55,'PEDI':35,'PUBH':35},
    'Медицинская кибернетика':       {'HELP':60,'EMPA':55,'STAB':70,'PREC':70,'MANUAL':30,'BIO_TOL':55,'TECH':90,'SCI':80,'PEDI':15,'PUBH':55},
    'Медицинская биофизика':         {'HELP':55,'EMPA':50,'STAB':70,'PREC':75,'MANUAL':30,'BIO_TOL':60,'TECH':80,'SCI':90,'PEDI':10,'PUBH':45},
}

CENTROIDS_SPO = {
    'Сестринское дело (СПО)':        {'HELP':80,'EMPA':75,'STAB':75,'PREC':70,'MANUAL':40,'BIO_TOL':70,'TECH':45,'SCI':45,'PEDI':30,'PUBH':45},
    'Лабораторная диагностика (СПО)':{'HELP':60,'EMPA':50,'STAB':70,'PREC':90,'MANUAL':35,'BIO_TOL':65,'TECH':75,'SCI':60,'PEDI':10,'PUBH':55},
    'Медицинский массаж (СПО)':      {'HELP':70,'EMPA':70,'STAB':70,'PREC':65,'MANUAL':85,'BIO_TOL':60,'TECH':35,'SCI':0, 'PEDI':0, 'PUBH':0},
    'Фармация (СПО)':                {'HELP':65,'EMPA':55,'STAB':70,'PREC':90,'MANUAL':35,'BIO_TOL':55,'TECH':70,'SCI':60,'PEDI':20,'PUBH':60},
}

# ----------------------------------------------------------
# ОПРОСНИК (30 пунктов)
# ----------------------------------------------------------
SCALES = {
    'HELP': [
        {"text":"Мне важно, чтобы моя работа приносила пользу людям.", "rev":False},
        {"text":"Мне приятно видеть, что благодаря мне человеку становится лучше.", "rev":False},
        {"text":"Я ощущаю смысл в том, чтобы посвятить жизнь помощи людям и развитию медицины.", "rev":False},
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
        {"text":"Мне интересно узнавать, как устроено тело человека и как работают лекарства.", "rev":False},
    ],
    'PEDI': [
        {"text":"Мне комфортно общаться с детьми.", "rev":False},
        {"text":"Терпеливо объясняю одно и то же ребёнку и его родителям.", "rev":False},
        {"text":"Мне нравится заботиться о младших.", "rev":False},
    ],
    'PUBH': [
        {"text":"Мне важно, чтобы соблюдались правила чистоты и безопасности.", "rev":False},
        {"text":"Меня интересуют вопросы здоровья всего сообщества.", "rev":False},
        {"text":"Мне близка идея предотвращать болезни, а не только лечить.", "rev":False},
    ],
}

ORDER = list(SCALES.keys())

# ----------------------------------------------------------
# ФУНКЦИИ
# ----------------------------------------------------------
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
    delta = IGMO_EXTRA_K * ((answers.get("HELP_2", 3) - 3) + (answers.get("SCI_2", 3) - 3))
    igmo = np.clip(base + delta, 0.0, 100.0)
    qc_flags = []
    vals=[]
    for sc,itms in SCALES.items():
        for i,it in enumerate(itms):
            v=answers[f"{sc}_{i}"]
            if it.get("rev"): v=reverse_score(v)
            vals.append(v)
    if np.std(vals)<0.5: qc_flags.append("Однообразные ответы")
    qc_penalty=min(IGMO_QC_MAX_PENALTY,3*len(qc_flags))
    return max(0.0, igmo-qc_penalty), fails, qc_flags

def save_to_csv(answers, profile, igmo, qc_flags, level):
    data = {"timestamp": datetime.datetime.now().isoformat(),
            "level": level, "IGMO": round(igmo,2),
            "QC_flags": "; ".join(qc_flags) if qc_flags else ""}
    for sc,val in profile.items(): data[sc]=round(val,2)
    df=pd.DataFrame([data])
    fn="results_krasgmu.csv"
    df.to_csv(fn, mode="a", header=not os.path.exists(fn),
              index=False, encoding="utf-8-sig")

def radar_chart(profile):
    labels = list(ORDER)
    values = [profile[k] for k in labels] + [profile[labels[0]]]
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist() + [0]
    fig = plt.figure(figsize=(6,6))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, values, color="tab:blue")
    ax.fill(angles, values, alpha=.15, color="tab:blue")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([SCALE_LABELS[k] for k in labels], fontsize=10)
    ax.tick_params(axis='x', pad=12)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_ylim(0, 100)
    ax.tick_params(pad=8)
    return fig

# --- Косинусное ранжирование ---
def _vec_from_profile(profile): return np.array([profile.get(k, 0.0) for k in SCALE_ORDER])
def _vec_from_centroid(centroid): return np.array([centroid[k] for k in SCALE_ORDER])
def _cosine(a,b):
    na=np.linalg.norm(a); nb=np.linalg.norm(b)
    return 0.0 if na==0 or nb==0 else float(np.dot(a,b)/(na*nb))
def rank_specialties(profile, level, fails):
    user=_vec_from_profile(profile)
    pool = CENTROIDS_VO if level.startswith("Высшего") else CENTROIDS_SPO
    ranks=[]
    for name, ctr in pool.items():
        sim=_cosine(user,_vec_from_centroid(ctr))
        score=sim*100.0*(GATE_PENALTY**len(fails))
        ranks.append((name,round(score,1)))
    ranks.sort(key=lambda x:x[1],reverse=True)
    return ranks
st.markdown("""
### Добро пожаловать!

Эта пилотная версия диагностической методики предназначеной для учащихся, которые интересуются медициной и хотят понять, какие направления обучения в КрасГМУ им наиболее подходят.

Методика **оценки готовности к медицинскому образованию (ИГМО)** позволяет определить, насколько ваши личностные особенности, интересы и склонности соответствуют различным медицинским специальностям.  
Результаты помогут:
- осознанно выбрать направление подготовки в медицине или смежных областях;
- увидеть собственные сильные стороны;
- понять, какие качества важно развивать для успешного обучения и будущей профессии.

Отвечайте искренне — **в тесте нет «правильных» и «неправильных» ответов**.  
Каждый пункт оценивается по шкале от 1 до 5, где  
**1 — не согласен**, а **5 — полностью согласен**.

Диагностика проводится анонимно. В конце вы увидите:
- **индекс готовности к медицинскому образованию (ИГМО)**;  
- **ваш психологический профиль**;
- и **ТОП-3 направлений КрасГМУ**, которые наиболее вам соответствуют.

После прочтения нажмите кнопку «Начать тестирование».
""")

if st.button("Начать тестирование"):
    st.session_state["started"] = True

if "started" not in st.session_state or not st.session_state["started"]:
    st.stop()

# ----------------------------------------------------------
# ОПРОСНИК
# ----------------------------------------------------------
st.markdown("### Уровень планируемого образования")
level = st.radio("Ты планируешь изучать медицину на уровне…",
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

# ----------------------------------------------------------
# ОБРАБОТКА РЕЗУЛЬТАТОВ
# ----------------------------------------------------------
if submitted:
    profile=compute_profile(answers)
    igmo,fails,qc_flags=compute_igmo(profile,answers,level)

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
        st.caption(f"Флаги качества: {', '.join(qc_flags)}")

    dfp=pd.DataFrame({"Шкала":[SCALE_LABELS[k] for k in ORDER],
                      "Баллы":[round(profile[k],1) for k in ORDER]})
    st.dataframe(dfp, use_container_width=True)
    st.pyplot(radar_chart(profile))

    save_to_csv(answers, profile, igmo, qc_flags, level)

    # --- РАНЖИРОВАНИЕ ---
    st.markdown("## Подходящие медицинские специальности")

    ranking = rank_specialties(profile, level, fails)
    top3 = ranking[:3]
    rest = ranking[3:]

    st.subheader("ТОП-3 подходящих специальностей")
    for name, sc in top3:
        st.write(f"**{name}** — {sc:.1f}%")
        st.progress(min(100, int(round(sc))))

    if rest:
        with st.expander("Показать остальные направления"):
            for name, sc in rest:
                st.write(f"{name} — {sc:.1f}%")

    st.caption("Результаты носят ориентировочный характер и не являются окончательным профессиональным заключением. Тест помогает лучше понять свои склонности, но не заменяет личную консультацию специалиста.")
