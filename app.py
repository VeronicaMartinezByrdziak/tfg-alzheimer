import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# =====================================================
# FUNCIÓN NECESARIA PARA CARGAR EL MODELO CLÍNICO
# =====================================================
def seleccionar_columnas_clinicas(X):
    return X


# =====================================================
# CONFIGURACIÓN
# =====================================================
st.set_page_config(
    page_title="Predicción de progresión a demencia",
    layout="centered"
)

st.title("Predicción de progresión a demencia")


# =====================================================
# ETIQUETAS BONITAS DEL MODELO CLÍNICO
# =====================================================
labels_clinico = {
    "ADAS13": "ADAS-13 – puntuación del test cognitivo (0–85)",
    "RAVLT_immediate": "RAVLT memoria inmediata – recuerdo verbal inmediato (0–75)",
    "RAVLT_perc_forgetting": "RAVLT porcentaje de olvido – información olvidada (%)",
    "FAQ": "FAQ – cuestionario de actividades funcionales (0–30)",
    "EcogSPMem": "ECog memoria – dificultades de memoria en la vida diaria (1–4)",
    "EcogSPPlan": "ECog planificación – capacidad de planificación en la vida diaria (1–4)",
    "EcogSPOrgan": "ECog organización – capacidad de organización en la vida diaria (1–4)",
    "EcogSPTotal": "ECog total – puntuación global de cognición diaria (1–4)",
    "mPACCtrailsB": "mPACC Trails-B – función ejecutiva (aprox. -15 a 5)"
}


# =====================================================
# ETIQUETAS BONITAS DEL MODELO BIOMÉDICO / ESTRUCTURAL
# =====================================================
labels_biomedico = {
    "FDG": "FDG-PET – metabolismo cerebral de glucosa (SUVR)",
    "AV45": "AV45-PET – carga cerebral de beta-amiloide (SUVR)",
    "ABETA": "Aβ42 en líquido cefalorraquídeo (pg/mL)",
    "PTAU": "p-tau en líquido cefalorraquídeo (pg/mL)",

    "ST111CV": "Volumen cortical del precúneo derecho (MRI, mm³)",
    "ST111TA": "Grosor cortical del precúneo derecho (MRI, mm)",
    "ST114TA": "Grosor cortical del frontal medio rostral derecho (MRI, mm)",
    "ST117TS": "Variabilidad del grosor cortical del temporal superior derecho (MRI, mm)",
    "ST12SV": "Volumen de la amígdala izquierda (MRI, mm³)",
    "ST13CV": "Volumen cortical del surco temporal superior izquierdo (MRI, mm³)",
    "ST25TA": "Grosor cortical del polo frontal izquierdo (MRI, mm)",
    "ST29SV": "Volumen del hipocampo izquierdo (MRI, mm³)",
    "ST40CV": "Volumen cortical del temporal medio izquierdo (MRI, mm³)",
    "ST58CV": "Volumen cortical del temporal superior izquierdo (MRI, mm³)",
    "ST60TS": "Variabilidad del grosor cortical del polo temporal izquierdo (MRI, mm)",
    "ST62TA": "Grosor cortical del giro temporal transverso izquierdo (MRI, mm)",
    "ST72CV": "Volumen cortical del surco temporal superior derecho (MRI, mm³)",
    "ST82TS": "Variabilidad del grosor cortical del cúneo derecho (MRI, mm)",
    "ST89SV": "Volumen del ventrículo lateral inferior derecho (MRI, mm³)",
    "ST90CV": "Volumen cortical del parietal inferior derecho (MRI, mm³)",
    "ST90TA": "Grosor cortical del parietal inferior derecho (MRI, mm)",
    "ST95TS": "Variabilidad del grosor cortical orbitofrontal derecho (MRI, mm)"
}


# =====================================================
# ETIQUETAS GENERALES
# =====================================================
labels_todas = {}
labels_todas.update(labels_clinico)
labels_todas.update(labels_biomedico)


# =====================================================
# BLOQUES DE VARIABLES DEL MODELO
# =====================================================
variables_clinicas_modelo = [
    "ADAS13", "RAVLT_immediate", "RAVLT_perc_forgetting", "FAQ",
    "EcogSPMem", "EcogSPPlan", "EcogSPOrgan", "EcogSPTotal", "mPACCtrailsB"
]

variables_lcr_modelo = ["ABETA", "PTAU"]
variables_pet_fdg_modelo = ["FDG"]
variables_pet_amiloide_modelo = ["AV45"]

variables_mri_modelo = [
    "ST111CV", "ST111TA", "ST114TA", "ST117TS", "ST12SV", "ST13CV",
    "ST25TA", "ST29SV", "ST40CV", "ST58CV", "ST60TS", "ST62TA",
    "ST72CV", "ST82TS", "ST89SV", "ST90CV", "ST90TA", "ST95TS"
]


# =====================================================
# VARIABLES CLAVE PARA SIMULACIÓN DE EMPEORAMIENTO
# =====================================================
variables_simulacion = {
    "FAQ": "sube",
    "EcogSPMem": "sube",
    "EcogSPTotal": "sube",
    "AV45": "sube",
    "PTAU": "sube",
    "ADAS13": "sube",
    "RAVLT_perc_forgetting": "sube",

    "ABETA": "baja",
    "ST29SV": "baja",
    "RAVLT_immediate": "baja",
    "FDG": "baja",
    "mPACCtrailsB": "baja"
}


# =====================================================
# CARGAR MODELOS
# =====================================================
@st.cache_resource
def cargar_modelos():
    modelo_clinico = joblib.load("modelo_clinico.joblib")
    modelo_biomedico = joblib.load("modelo_alzheimer.joblib")
    return modelo_clinico, modelo_biomedico


modelo_clinico, modelo_biomedico = cargar_modelos()


# =====================================================
# CONTROL DE PANTALLA
# =====================================================
if "pantalla" not in st.session_state:
    st.session_state.pantalla = "inicio"

if "modelo" not in st.session_state:
    st.session_state.modelo = None


# =====================================================
# FUNCIÓN AUXILIAR PARA LIMPIAR ENTRADAS
# =====================================================
def limpiar_valor_entrada(valor):
    if valor is None:
        return np.nan

    valor = str(valor).strip()

    if valor == "":
        return np.nan

    valor = valor.replace(",", ".")
    return pd.to_numeric(valor, errors="coerce")


# =====================================================
# VARIABLES FALTANTES
# =====================================================
def obtener_variables_faltantes(df, columnas_modelo):
    faltantes = []
    for col in columnas_modelo:
        if pd.isna(df.at[0, col]):
            faltantes.append(col)
    return faltantes


# =====================================================
# MENSAJE CORTO DE PRUEBAS RECOMENDADAS
# =====================================================
def obtener_pruebas_recomendadas(faltantes):
    pruebas = []

    faltan_clinicas = any(v in faltantes for v in variables_clinicas_modelo)
    faltan_lcr = any(v in faltantes for v in variables_lcr_modelo)
    faltan_fdg = any(v in faltantes for v in variables_pet_fdg_modelo)
    faltan_av45 = any(v in faltantes for v in variables_pet_amiloide_modelo)
    faltan_mri = any(v in faltantes for v in variables_mri_modelo)

    if faltan_clinicas:
        pruebas.append("evaluación cognitiva y funcional")
    if faltan_mri:
        pruebas.append("resonancia magnética cerebral")
    if faltan_lcr:
        pruebas.append("biomarcadores en líquido cefalorraquídeo")
    if faltan_fdg:
        pruebas.append("FDG-PET")
    if faltan_av45:
        pruebas.append("AV45-PET")

    return pruebas


def mostrar_aviso_pruebas_recomendadas(faltantes):
    pruebas = obtener_pruebas_recomendadas(faltantes)

    if len(pruebas) == 0:
        return

    texto = "Pruebas recomendadas: " + ", ".join(pruebas) + "."
    st.info(texto)


# =====================================================
# FUNCIÓN PARA SIMULAR EMPEORAMIENTO Y GRAFICAR EL RIESGO
# =====================================================
def graficar_simulacion_empeoramiento(df_base, pipe, columnas_modelo, threshold):
    niveles = [0, 5, 10, 15, 20, 25]
    riesgos = []

    variables_usadas = []
    for var, direccion in variables_simulacion.items():
        if var in df_base.columns and pd.notna(df_base.at[0, var]):
            variables_usadas.append((var, direccion))

    if len(variables_usadas) == 0:
        st.info("No hay suficientes variables clave completadas para mostrar la simulación de empeoramiento.")
        return

    for nivel in niveles:
        df_sim = df_base.copy()
        factor = nivel / 100.0

        for var, direccion in variables_usadas:
            valor_original = float(df_base.at[0, var])

            if direccion == "sube":
                if valor_original == 0:
                    nuevo_valor = valor_original + factor
                else:
                    nuevo_valor = valor_original * (1 + factor)
            else:
                nuevo_valor = valor_original * (1 - factor)

            df_sim.at[0, var] = nuevo_valor

        proba_sim = pipe.predict_proba(df_sim[columnas_modelo])[:, 1][0]
        riesgos.append(proba_sim * 100)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(niveles, riesgos, marker="o")
    ax.axhline(threshold * 100, linestyle="--", label="Umbral del modelo")
    ax.set_xlabel("Empeoramiento simulado aplicado a variables clave (%)")
    ax.set_ylabel("Riesgo predicho (%)")
    ax.set_title("Simulación de empeoramiento y cambio del riesgo")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend()

    st.pyplot(fig)

    nombres_bonitos = [labels_todas.get(var, var) for var, _ in variables_usadas]
    st.caption("Variables usadas en la simulación: " + "; ".join(nombres_bonitos))
    st.caption(
        "Esta gráfica muestra cómo cambia la salida del modelo si empeoran de forma simulada "
        "algunas variables clave seleccionadas a partir del análisis SHAP. "
        "No equivale a una evolución temporal real del paciente."
    )


# =====================================================
# PANTALLA 1 — SELECCIÓN DE MODELO
# =====================================================
if st.session_state.pantalla == "inicio":

    st.header("Selecciona el tipo de modelo")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Modelo clínico"):
            st.session_state.modelo = "clinico"
            st.session_state.pantalla = "prediccion"
            st.rerun()

    with col2:
        if st.button("Modelo clínico + biomédico"):
            st.session_state.modelo = "biomedico"
            st.session_state.pantalla = "prediccion"
            st.rerun()


# =====================================================
# PANTALLA 2 — PREDICCIÓN
# =====================================================
if st.session_state.pantalla == "prediccion":

    if st.button("⬅ Volver"):
        st.session_state.pantalla = "inicio"
        st.rerun()

    if st.session_state.modelo == "clinico":
        artefacto = modelo_clinico
        st.subheader("Modelo clínico")
    else:
        artefacto = modelo_biomedico
        st.subheader("Modelo clínico + biomédico")

    pipe = artefacto["pipeline"]
    columnas = artefacto["columnas_modelo"]
    threshold = artefacto["threshold"]

    st.write("Introduce los datos del paciente")
    st.caption("Puedes dejar campos vacíos. También puedes escribir decimales con coma o con punto.")

    datos = {}

    for col in columnas:
        etiqueta = labels_todas.get(col, col)
        datos[col] = st.text_input(etiqueta, key=f"input_{col}")

    if st.button("Predecir"):

        rellenadas = sum(1 for v in datos.values() if str(v).strip() != "")
        total = len(columnas)
        porcentaje = rellenadas / total

        st.write(f"Variables introducidas: {rellenadas}/{total} ({porcentaje*100:.1f}%)")

        df = pd.DataFrame([datos])

        for c in df.columns:
            df[c] = df[c].apply(limpiar_valor_entrada)

        df = df[columnas]
        faltantes = obtener_variables_faltantes(df, columnas)

        # =====================================================
        # MUY POCOS DATOS: NO PREDICCIÓN
        # =====================================================
        if porcentaje < 0.40:
            st.error("No hay suficientes datos para realizar una predicción fiable.")
            mostrar_aviso_pruebas_recomendadas(faltantes)
            st.stop()

        # =====================================================
        # PREDICCIÓN
        # =====================================================
        proba = pipe.predict_proba(df)[:, 1][0]
        pred = int(proba >= threshold)

        st.write(f"Probabilidad: {proba:.3f}")

        # =====================================================
        # DATOS INCOMPLETOS PERO SUFICIENTES PARA PREDECIR
        # =====================================================
        if porcentaje < 0.60:
            st.warning(
                "La predicción se ha realizado con datos incompletos."
            )

            margen_incertidumbre = 0.05
            caso_incierto = abs(proba - threshold) < margen_incertidumbre

            if pred == 1:
                st.error("Alto riesgo de progresión.")
                mostrar_aviso_pruebas_recomendadas(faltantes)
            else:
                st.success("Bajo riesgo de progresión.")

                if caso_incierto:
                    mostrar_aviso_pruebas_recomendadas(faltantes)
        else:
            if pred == 1:
                st.error("Alto riesgo de progresión.")
            else:
                st.success("Bajo riesgo de progresión.")

        # =====================================================
        # GRÁFICA SOLO EN MODELO CLÍNICO + BIOMÉDICO
        # =====================================================
        if st.session_state.modelo == "biomedico":
            st.subheader("Simulación de empeoramiento")
            st.info(
                "La siguiente gráfica muestra cómo cambia el riesgo predicho si empeoran de forma simulada "
                "algunas de las variables más influyentes del modelo, seleccionadas a partir del análisis SHAP."
            )
            graficar_simulacion_empeoramiento(df, pipe, columnas, threshold)