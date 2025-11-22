import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt

# FUNCIÓN PARA CREAR EL GRAFO
def crear_grafo_libre_de_escala(n_nodos, m_conexiones):
   
    G = nx.barabasi_albert_graph(n_nodos, m_conexiones)
    return G

# FUNCIÓN PARA EL ALGORITMO FAST-SIR 
def simular_fast_sir(G, beta, gamma, nodo_inicial=None):
    """
    Simula la propagación SIR en un grafo G usando la lógica de FastSIR.

    G: El grafo de NetworkX.
    beta: Tasa de infección.
    gamma: Tasa de recuperación.
    nodo_inicial: El nodo donde inicia la propagación.
    """
    n_nodos = G.number_of_nodes()

    # Inicialización de estados: S=0, I=1, R=2
    estado = np.zeros(n_nodos, dtype=int) # Todos son Susceptibles (S=0)
    
    # 1. Elegir el nodo inicial 'Infectado' (I=1)
    if nodo_inicial is None:
        inicio = random.choice(list(G.nodes()))
    else:
        inicio = nodo_inicial

    estado[inicio] = 1 # El nodo inicial se infecta
    infectados_activos = {inicio}
    
    # Listas para almacenar la evolución (para el gráfico S-I-R)
    S_hist, I_hist, R_hist = [n_nodos - 1], [1], [0]

    # Probabilidad de infección P_ij basada en FastSIR (probabilidad de transmisión)
    prob_transmision = beta / (beta + gamma)

    
    while infectados_activos:
        # Los nodos infectados en esta 'generación'
        infectados_actuales = list(infectados_activos)
        nuevos_infectados = set()
        nodos_recuperados = set()

        # 2. PROCESO DE CONTAGIO
        for i in infectados_actuales:
            # Los infectados intentan contagiar a sus vecinos S
            for vecino in G.neighbors(i):
                if estado[vecino] == 0: # Si es Susceptible
                    # Determinamos si el contagio ocurre con la probabilidad P_ij
                    if np.random.rand() < prob_transmision:
                        nuevos_infectados.add(vecino)
            
            # 3. PROCESO DE RECUPERACIÓN
            # En el modelo FastSIR, el nodo se recupera *después* de intentar la transmisión
            nodos_recuperados.add(i)

        # 4. ACTUALIZAR ESTADOS
        for j in nuevos_infectados:
            if estado[j] == 0: # Solo si no ha sido infectado ya
                estado[j] = 1 # Pasa a Infectado (I=1)
                infectados_activos.add(j)

        for k in nodos_recuperados:
            if estado[k] == 1: # Solo si sigue Infectado (para evitar re-recuperación)
                estado[k] = 2 # Pasa a Recuperado (R=2)
                infectados_activos.discard(k) # Lo quitamos de los activos
        
        # 5. Guardar la evolución para la gráfica S-I-R
        S_t = np.sum(estado == 0)
        I_t = np.sum(estado == 1)
        R_t = np.sum(estado == 2)
        S_hist.append(S_t)
        I_hist.append(I_t)
        R_hist.append(R_t)
        
    return S_hist, I_hist, R_hist, estado

# FUNCIÓN PRINCIPAL DE EJECUCIÓN Y VISUALIZACIÓN 
def ejecutar_simulacion_y_graficar(n_nodos, beta, gamma):
    # Parámetros del Grafo BA
    m_conexiones = 2 
    
    # 1. Crear el grafo
    G = crear_grafo_libre_de_escala(n_nodos, m_conexiones)
    print(f"Grafo creado con {G.number_of_nodes()} nodos y {G.number_of_edges()} aristas.")
    
    # 2. Simular
    S_hist, I_hist, R_hist, estado_final = simular_fast_sir(G, beta, gamma)
    
    # 3. Mostrar resultados y métricas
    print("\n--- Resultados de la Simulación ---")
    print(f"Tasa de Infección (β): {beta:.3f}")
    print(f"Tasa de Recuperación (γ): {gamma:.3f}")
    print(f"Duración de la simulación (generaciones): {len(S_hist) - 1}")
    print(f"Alcance Final (Recuperados): {R_hist[-1]} nodos ({R_hist[-1]/n_nodos*100:.2f}%)")
    
    # 4. Graficar la evolución S-I-R
    plt.figure(figsize=(10, 6))
    plt.plot(S_hist, label='Susceptible (S)', color='blue')
    plt.plot(I_hist, label='Infectado (I)', color='red')
    plt.plot(R_hist, label='Recuperado/Retirado (R)', color='green')
    
    plt.title('Propagación de Noticias Falsas (Modelo SIR)')
    plt.xlabel('Generaciones de Propagación')
    plt.ylabel('Número de Usuarios')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()



# FUNCIÓN PARA ANALIZAR EL IMPACTO DEL NODO INICIAL
def analizar_impacto_nodo_inicial(n_nodos, beta, gamma):
    """
    Compara el Alcance Final de la simulación FastSIR iniciando desde un Hub 
    (alto grado) vs. un Periférico (bajo grado).
    """
    
    m_conexiones = 2
    G = crear_grafo_libre_de_escala(n_nodos, m_conexiones)
    
    # 1. IDENTIFICAR LOS NODOS CLAVE
    # Obtener el grado de todos los nodos
    grados = dict(G.degree())
    
    # Encontrar el Hub (nodo con el mayor grado)
    hub_nodo = max(grados, key=grados.get)
    
    # Encontrar el Periférico (nodo con el menor grado, que no sea 0 si la red es conexa)
    periferico_nodo = min(grados, key=grados.get)
    
    puntos_de_inicio = {
        f"Hub": hub_nodo,
        f"Periférico": periferico_nodo
    }
    
    alcances_finales = {}
    num_repeticiones = 10 # Promediamos para reducir el ruido estocástico

    print(f"\n--- Análisis: Influencia del Punto de Inicio ---")
    print(f"Beta: {beta}, Gamma: {gamma}")

    # 2. SIMULACIÓN Y PROMEDIADO
    for nombre_inicio, nodo_inicio in puntos_de_inicio.items():
        R_final_promedio = 0
        
        for _ in range(num_repeticiones):
            # Usamos el nodo_inicio específico en la simulación
            _, _, R_hist, _ = simular_fast_sir(G, beta, gamma, nodo_inicial=nodo_inicio)
            R_final_promedio += R_hist[-1]
            
        R_final_promedio /= num_repeticiones
        
        alcance_porcentaje = (R_final_promedio / n_nodos) * 100
        alcances_finales[nombre_inicio] = alcance_porcentaje
        print(f"  Inicio en {nombre_inicio}: Alcance Final Promedio: {alcance_porcentaje:.2f}%")

    # 3. GRAFICAR LOS RESULTADOS (Gráfico de Barras)
    nombres = list(alcances_finales.keys())
    valores = list(alcances_finales.values())

    plt.figure(figsize=(9, 6))
    plt.bar(nombres, valores, color=['green', 'orange'])
    
    plt.title('Impacto del Nodo Inicial en el Alcance Final de la Propagación')
    plt.xlabel('Tipo de Usuario Inicial')
    plt.ylabel('Porcentaje de Alcance Final de la Noticia Falsa (%)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.ylim(0, max(valores) * 1.2)
    plt.show()

    return alcances_finales

# FUNCIÓN PARA ANÁLISIS DE SENSIBILIDAD DE GAMMA 
def analizar_sensibilidad_gamma(n_nodos, beta, gamma_min, gamma_max, num_pasos):
    """
    Realiza múltiples simulaciones variando la tasa de recuperación (gamma).
    
    n_nodos: Número de nodos en la red.
    beta: Tasa de infección (fija).
    gamma_min, gamma_max: Rango de gamma a probar.
    num_pasos: Número de puntos a muestrear en el rango.
    """
    
    # 1. Crear el grafo
    G = crear_grafo_libre_de_escala(n_nodos, m_conexiones=2)
    print(f"\n--- Análisis de Sensibilidad (γ) ---")
    print(f"Grafo de {n_nodos} nodos creado. Beta fija: {beta}")
    
    # Rango de gammas a probar
    gammas = np.linspace(gamma_min, gamma_max, num_pasos)
    alcances_finales = []

    # 2. Iterar sobre los valores de gamma
    for gamma in gammas:
        # Ejecutar la simulación FastSIR
        # Nota: Usamos 3 repeticiones por punto para promediar el efecto estocástico
        R_final_promedio = 0
        num_repeticiones = 3 
        
        for _ in range(num_repeticiones):
            # No necesitamos las historias S, I, R, solo el estado final
            _, _, R_hist, _ = simular_fast_sir(G, beta, gamma)
            R_final_promedio += R_hist[-1]
            
        R_final_promedio /= num_repeticiones
        
        # Almacenar el resultado (porcentaje de alcance final)
        alcance_porcentaje = (R_final_promedio / n_nodos) * 100
        alcances_finales.append(alcance_porcentaje)
        print(f"  γ={gamma:.3f} | Alcance Final: {alcance_porcentaje:.2f}%")

    # 3. Graficar los resultados
    plt.figure(figsize=(10, 6))
    plt.plot(gammas, alcances_finales, marker='o', linestyle='-', color='purple')
    
    plt.title('Impacto de la Tasa de Recuperación (γ) en el Alcance Final')
    plt.xlabel('Tasa de Recuperación (γ)')
    plt.ylabel('Porcentaje de Alcance Final de la Noticia Falsa (%)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

    return gammas, alcances_finales


# FUNCIÓN PARA CREAR EL GRAFO ALEATORIO 
def crear_grafo_aleatorio(n_nodos, prob_conexion):
    """
    Crea un grafo aleatorio (Erdos-Renyi).
    Cada par de nodos tiene una probabilidad 'prob_conexion' de estar conectado.
    """
    G = nx.erdos_renyi_graph(n_nodos, prob_conexion)
    return G

# FUNCIÓN PARA COMPARAR TOPOLOGÍAS 
def comparar_topologias(n_nodos, beta, gamma):
    """
    Compara el Alcance Final de la simulación FastSIR en dos tipos de grafos.
    """
    
    # 1. Parámetros de la Red
    m_conexiones_ba = 2 # Parámetro para Barabási-Albert
    
    # Estimamos la probabilidad para Erdos-Renyi para tener un grado promedio similar:
    # Grado promedio ~ 2 * m_conexiones_ba = 4
    # En Erdos-Renyi, grado promedio ~ (N-1) * p. Si N=5000, p ~ 4/5000 = 0.0008
    prob_er = 4 / n_nodos # Probabilidad de conexión para Erdos-Renyi
    
    # 2. Creación de Grafos
    G_BA = crear_grafo_libre_de_escala(n_nodos, m_conexiones_ba)
    G_ER = crear_grafo_aleatorio(n_nodos, prob_er)
    
    tipos_de_grafo = {
        "Libre de Escala (BA)": G_BA,
        "Aleatorio (ER)": G_ER
    }
    
    alcances_finales = {}
    num_repeticiones = 5 # Más repeticiones para promediar la estocasticidad

    print(f"\n--- Comparación de Topologías de Red ---")
    print(f"Beta: {beta}, Gamma: {gamma}. Repeticiones por grafo: {num_repeticiones}")

    # 3. Simulación y Promediado
    for nombre, G in tipos_de_grafo.items():
        R_final_promedio = 0
        
        for _ in range(num_repeticiones):
            # Ejecutar FastSIR
            _, _, R_hist, _ = simular_fast_sir(G, beta, gamma)
            R_final_promedio += R_hist[-1]
            
        R_final_promedio /= num_repeticiones
        
        alcance_porcentaje = (R_final_promedio / n_nodos) * 100
        alcances_finales[nombre] = alcance_porcentaje
        print(f"  Grafo {nombre}: Alcance Final Promedio: {alcance_porcentaje:.2f}%")

    # 4. Graficar los resultados (Gráfico de Barras)
    nombres = list(alcances_finales.keys())
    valores = list(alcances_finales.values())

    plt.figure(figsize=(8, 6))
    plt.bar(nombres, valores, color=['red', 'blue'])
    
    plt.title('Alcance Final de la Noticia Falsa por Tipo de Red')
    plt.xlabel('Tipo de Grafo (Estructura de la Red)')
    plt.ylabel('Porcentaje de Alcance Final (%)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.ylim(0, max(valores) * 1.2) # Ajustar el límite Y para mejor visualización
    plt.show()

    return alcances_finales

#parte 2
# FUNCIÓN PARA VISUALIZAR EL INICIO DEL GRAFO 
def visualizar_grafo_inicial(n_nodos_ejemplo, beta, gamma):
    """
    Crea y visualiza el estado inicial de un grafo.
    """
    m_conexiones = 2
    G = crear_grafo_libre_de_escala(n_nodos_ejemplo, m_conexiones)
    
    # Inicialización de estados: S=0, I=1, R=2
    n_nodos = G.number_of_nodes()
    estado_inicial = np.zeros(n_nodos, dtype=int) # Todos son Susceptibles (S=0)
    
    # Elegir el nodo inicial 'Infectado' (I=1)
    inicio = random.choice(list(G.nodes()))
    estado_inicial[inicio] = 1 # El nodo inicial se infecta
    
    # 1. ASIGNAR COLORES
    colores = []
    mapa_colores = {0: 'blue', 1: 'red', 2: 'green'} # R=2 (verde) no se usa aquí
    
    for nodo in G.nodes():
        estado = estado_inicial[nodo]
        colores.append(mapa_colores[estado])
        
    # 2. GRAFICAR EL GRAFO
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42, k=0.3) 
    
    nx.draw_networkx_nodes(G, pos, node_color=colores, node_size=100, alpha=0.8)
    nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.4)
    
    # Crear leyenda manualmente
    plt.scatter([], [], c='blue', s=150, label='Susceptible (S)')
    plt.scatter([], [], c='red', s=150, label='Infectado (I)')
    plt.scatter([], [], c='green', s=150, label='Recuperado (R)')

    plt.title(f'Visualización de la Red: Estado INICIAL (N={n_nodos_ejemplo})', fontsize=15)
    plt.legend(scatterpoints=1)
    plt.axis('off')
    plt.show()


# FUNCIÓN PARA VISUALIZAR EL FIN DEL GRAFO 
def visualizar_grafo_final(n_nodos_ejemplo, beta, gamma):
    """
    Crea, simula y visualiza un grafo más pequeño con colores según el estado final (S, I, R).
    """
    m_conexiones = 2
    G = crear_grafo_libre_de_escala(n_nodos_ejemplo, m_conexiones)
    
    # Simular la propagación
    _, _, _, estado_final = simular_fast_sir(G, beta, gamma)
    
    # 1. ASIGNAR COLORES
    # S=0 (Azul), I=1 (Rojo), R=2 (Verde)
    colores = []
    
    # Mapeo de estados a colores
    mapa_colores = {0: 'blue', 1: 'red', 2: 'green'}
    
    for nodo in G.nodes():
        estado = estado_final[nodo]
        colores.append(mapa_colores[estado])
        
    # 2. GRAFICAR EL GRAFO
    plt.figure(figsize=(10, 8))
    
    # Usamos un layout (por ejemplo, spring_layout) para posicionar los nodos
    pos = nx.spring_layout(G, seed=42, k=0.3) 
    
    nx.draw_networkx_nodes(G, pos, node_color=colores, node_size=100, alpha=0.8)
    nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.4)
    
    # Crear leyenda manualmente
    for estado, color in mapa_colores.items():
        if estado == 0: etiqueta = 'Susceptible (S)'
        elif estado == 1: etiqueta = 'Infectado (I)'
        else: etiqueta = 'Recuperado (R)'
        plt.scatter([], [], c=color, s=150, label=etiqueta)

    plt.title(f'Visualización de la Propagación Final (N={n_nodos_ejemplo}, β={beta}, γ={gamma})', fontsize=15)
    plt.legend(scatterpoints=1)
    plt.axis('off') # Ocultar ejes
    plt.show()


# CONFIGURACIÓN INICIAL
if __name__ == '__main__':
    # Configuración de la Red
    NUM_NODOS = 5000  # Número total de usuarios en la red
    BETA = 0.5        # Tasa de infección
    GAMMA = 0.3       # Tasa de recuperación
    
    # Ejecutar la simulación base (Curvas S-I-R)
    ejecutar_simulacion_y_graficar(NUM_NODOS, BETA, GAMMA)
    
    # Ejecutar el Análisis de Sensibilidad (γ) 
    GAMMA_MIN = 0.1
    GAMMA_MAX = 1.5
    NUM_PASOS = 15
    
    analizar_sensibilidad_gamma(
        NUM_NODOS, 
        BETA, 
        GAMMA_MIN, 
        GAMMA_MAX, 
        NUM_PASOS
    )

    # Ejecutar la Comparación de Topologías 
    comparar_topologias(NUM_NODOS, BETA, GAMMA)

    
    # Ejecutar el Análisis del Impacto del Nodo Inicial 
    analizar_impacto_nodo_inicial(
        NUM_NODOS, 
        BETA, 
        GAMMA
    )
    
    # Ejecutar la Visualización del Grafo 
    # Usamos una red mucho más pequeña para que se pueda dibujar
    NODOS_VISUALIZACION = 100 

    visualizar_grafo_inicial(
            NODOS_VISUALIZACION, 
            BETA, 
            GAMMA
        )
    
    visualizar_grafo_final(
        NODOS_VISUALIZACION, 
        BETA, 
        GAMMA
    )
