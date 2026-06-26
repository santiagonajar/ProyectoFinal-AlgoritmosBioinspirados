import numpy as np
import matplotlib.pyplot as plt


# CONFIGURACIÓN DE LOS PROBLEMAS DE OPTIMIZACIÓN CON RESTRICCIONES

def resorte_objetivo(t):
    """
    Función objetivo para el problema del diseño de un resorte de tensión/compresión.
    Minimiza el peso del resorte.
    t = [t1, t2, t3] -> [diámetro del alambre (d), diámetro medio de la bobina (D), número de bobinas activas (N)]
    """
    return (t[2] + 2) * t[1] * (t[0]**2)

def resorte_restricciones(t):
    """
    Calcula las restricciones del problema del resorte.
    Retorna un arreglo con el nivel de violación de cada restricción (max(0, g(x))).
    Un valor de 0 indica que la restricción se cumple (solución factible en ese parámetro).
    """
    t1, t2, t3 = t
    g = [
        1 - (t2**3 * t3) / (71785 * t1**4),
        (4 * t2**2 - t1 * t2) / (12566 * (t2 * t1**3 - t1**4)) + 1 / (5108 * t1**2) - 1,
        1 - (140.45) / (t2**2 * t3),
        (t1 + t2) / 1.5 - 1
    ]
    return np.array([max(0, val) for val in g])

def viga_objetivo(x):
    """
    Función objetivo para el problema de la viga soldada.
    Minimiza el costo de fabricación de la viga.
    x = [x1, x2, x3, x4] -> [espesor de la soldadura (h), longitud del soporte (l), altura de la viga (t), espesor de la viga (b)]
    """
    return 1.10471 * x[0]**2 * x[1] + 0.04811 * x[2] * x[3] * (14.0 + x[1])

def viga_restricciones(x):
    """
    Calcula las restricciones de diseño mecánico para la viga soldada (esfuerzos, flexión, pandeo, etc.).
    Retorna un arreglo con el nivel de violación de cada restricción.
    """
    x1, x2, x3, x4 = x 
    P, L, E, G = 6000, 14, 30e6, 12e6
    t_max, s_max, d_max = 13600, 30000, 0.25
    
    # Cálculos mecánicos intermedios
    M = P * (L + x2/2)
    R = np.sqrt(x2**2/4 + (x1+x3)**2/4)
    J = 2 * (np.sqrt(2) * x1 * x2 * (x2**2/12 + (x1+x3)**2/4))
    tp = P / (np.sqrt(2) * x1 * x2)
    tpp = M * R / J
    tau = np.sqrt(tp**2 + 2*tp*tpp*(x2/(2*R)) + tpp**2)
    sigma = (6 * P * L) / (x4 * x3**2)
    delta = (4 * P * L**3) / (E * x3**3 * x4)
    Pc = (4.013 * E * np.sqrt((x3**2 * x4**6)/36) / L**2) * (1 - (x3/(2*L)) * np.sqrt(E/(4*G)))
    
    g = [
        tau - t_max,
        sigma - s_max,
        x1 - x4, 
        1.10471 * x1**2 * x2 + 0.04811 * x3 * x4 * (14.0 + x2) - 5.0,
        0.125 - x1,
        delta - d_max,
        P - Pc
    ]
    return np.array([max(0, val) for val in g])


# MANEJO DE RESTRICCIONES POR STOCHASTIC RANKING PARA UNA PAREJA

def comparar_stochastic_ranking(f1, phi1, f2, phi2, Pf):
    """
    Aplica el criterio de Stochastic Ranking para determinar si el candidato 2
    es mejor que el candidato 1 bajo una probabilidad Pf de priorizar la función objetivo.
    
    Retorna True si el candidato 2 es mejor (se acepta el reemplazo), False de lo contrario.
    """
    u = np.random.rand()
    # Si ambos son factibles o por azar (u < Pf), comparamos únicamente por función objetivo
    if (phi1 == 0 and phi2 == 0) or (u < Pf):
        return f2 < f1
    else:
        # Si no, comparamos por la suma total del cuadrado de violación de restricciones
        return phi2 < phi1


# METAHEURÍSTICA: RECOCIDO SIMULADO CON MANEJO DE RESTRICCIONES

def recocido_simulado_restricciones(obj_func, rest_func, bounds, Pf, 
                                   t_inicial=100.0, t_final=0.001, 
                                   factor_enfriamiento=0.95, iter_por_temp=50):
    """
    Implementación del algoritmo de Recocido Simulado adaptado a optimización 
    con restricciones mediante Stochastic Ranking y Criterio de Metrópolis modificado.
    """
    dim = len(bounds)
    
    # 1. Generación de una solución inicial aleatoria dentro de los límites
    sol_actual = np.random.uniform(bounds[:, 0], bounds[:, 1], dim)
    f_actual = obj_func(sol_actual)
    phi_actual = np.sum(rest_func(sol_actual)**2)
    
    # Inicializamos el registro de la mejor solución histórica hallada
    mejor_sol = np.copy(sol_actual)
    mejor_f = f_actual
    mejor_phi = phi_actual
    
    t_actual = t_inicial
    history = [] # Guardará tuplas de (Mejor_Función_Objetivo, Es_Factible) por iteración térmica
    
    # 2. Ciclo principal de enfriamiento
    while t_actual > t_final:
        for _ in range(iter_por_temp):
            # Generación de una solución vecina mediante perturbación gaussiana adaptativa
            amplitud = (bounds[:, 1] - bounds[:, 0]) * 0.05
            vecino = sol_actual + np.random.normal(0, amplitud, dim)
            vecino = np.clip(vecino, bounds[:, 0], bounds[:, 1]) # Retener dentro de la caja de búsqueda
            
            f_vecino = obj_func(vecino)
            phi_vecino = np.sum(rest_func(vecino)**2)
            
            # Evaluamos si el vecino es mejor según las reglas de Stochastic Ranking
            es_mejor = comparar_stochastic_ranking(f_actual, phi_actual, f_vecino, phi_vecino, Pf)
            
            if es_mejor:
                # Aceptación directa si es mejor de acuerdo al criterio seleccionado
                sol_actual, f_actual, phi_actual = vecino, f_vecino, phi_vecino
            else:
                # Si no es mejor, calculamos un delta de energía virtual para aplicar el criterio de Metrópolis.
                # Se penaliza la energía si la solución es infactible para desalentar quedarse ahí.
                energia_actual = f_actual + (phi_actual * 1e5 if phi_actual > 0 else 0)
                energia_vecino = f_vecino + (phi_vecino * 1e5 if phi_vecino > 0 else 0)
                delta_e = energia_vecino - energia_actual
                
                # Criterio probabilístico de Metrópolis
                if delta_e < 0 or np.random.rand() < np.exp(-delta_e / t_actual):
                    sol_actual, f_actual, phi_actual = vecino, f_vecino, phi_vecino
            
            # Actualización de la mejor solución global encontrada en la ejecución
            # Optamos por soluciones factibles o que minimicen severamente la infactibilidad
            if phi_actual == 0 and (mejor_phi > 0 or f_actual < mejor_f):
                mejor_sol, mejor_f, mejor_phi = np.copy(sol_actual), f_actual, phi_actual
            elif mejor_phi > 0 and phi_actual < mejor_phi:
                mejor_sol, mejor_f, mejor_phi = np.copy(sol_actual), f_actual, phi_actual

        # Guardar estado actual del ciclo térmico para la gráfica de convergencia
        # MODIFICACIÓN: Guardamos tupla extendida incluyendo el valor instantáneo actual para graficación avanzada sin romper la firma del método
        history.append((mejor_f, mejor_phi == 0, f_actual, phi_actual == 0))
        
        # Enfriamiento geométrico del sistema
        t_actual *= factor_enfriamiento
        
    return history, (1 if mejor_phi == 0 else 0), mejor_f


# PLANTILLA DE EJECUCIÓN EXPERIMENTAL Y GENERACIÓN DE REPORTES

def graficar_convergencia(p_name, histories, p_f_list):
    """
    Genera los gráficos de convergencia requeridos para visualizar la evolución del
    algoritmo comparando los distintos valores del parámetro Pf.
    """
    plt.figure(figsize=(10, 6))
    for i, pf in enumerate(p_f_list):
        hist = histories[i]
        vals = [h[0] for h in hist]
        fact = [h[1] for h in hist]
        
        plt.plot(vals, label=f'Pf = {pf}', linewidth=2)
        
        # Añadir un marcador visual 'x' de color rojo cuando el mejor individuo de esa iteración sea infactible
        inf_x = [g for g, f in enumerate(fact) if not f]
        inf_y = [vals[g] for g in inf_x]
        if inf_x:
            plt.scatter(inf_x, inf_y, marker='x', color='red', s=25, alpha=0.6)

    plt.title(f'Convergencia de Recocido Simulado: {p_name.capitalize()}')
    plt.xlabel('Iteraciones de Enfriamiento')
    plt.ylabel('Mejor Valor de la Función Objetivo')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()


def graficar_comportamiento_instantaneo(p_name, histories, p_f_list):
    """
    Genera gráficos adicionales que contrastan el comportamiento caótico instantáneo
    (subidas y bajadas del Criterio de Metrópolis) frente a la bitácora del óptimo histórico.
    """
    for i, pf in enumerate(p_f_list):
        hist = histories[i]
        mejor_vals = [h[0] for h in hist]
        inst_vals = [h[2] for h in hist]
        
        plt.figure(figsize=(11, 4.5))
        plt.plot(inst_vals, color='blue', alpha=0.75, label='Valor Instantáneo Actual (Exploración de Metrópolis)', linewidth=1)
        plt.plot(mejor_vals, color='red', label='Bitácora Histórica (Óptimo Acumulado)', linewidth=2.5)
        
        plt.title(f'Exploración Térmica Instantánea ({p_name.capitalize()}) - Parámetro Pf = {pf}')
        plt.xlabel('Iteraciones de Enfriamiento')
        plt.ylabel('Valor de la Función Objetivo')
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.show()

# Parámetros del espacio de búsqueda (Caja de restricciones)
params = {
    'resorte': {
        'func': resorte_objetivo, 
        'res': resorte_restricciones, 
        'b': np.array([[0.05, 2.0], [0.25, 1.3], [2.0, 15.0]])
    },
    'viga': {
        'func': viga_objetivo, 
        'res': viga_restricciones, 
        'b': np.array([[0.1, 2.0], [0.1, 10.0], [0.1, 10.0], [0.1, 2.0]])
    }
}

p_f_values = [0.2, 0.45, 0.8]
all_results = []

print("Iniciando ejecuciones experimentales de Recocido Simulado...\n")

for p_name, p_data in params.items():
    histories_to_plot = []
    
    for pf in p_f_values:
        runs_best = []
        runs_fact = []
        best_history_run = None
        min_f_found = float('inf')
        
        # Ejecución de 30 corridas estadísticas independientes por cada combinación para obtener la mediana
        for s in range(30):
            np.random.seed(s) # Asegurar reproducibilidad de los experimentos
            
            h, f_status, final_f = recocido_simulado_restricciones(
                p_data['func'], p_data['res'], p_data['b'], pf,
                t_inicial=150.0, t_final=0.005, factor_enfriamiento=0.93, iter_por_temp=40
            )
            
            runs_best.append(final_f)
            runs_fact.append(f_status)
            
            # Guardamos el historial que represente el comportamiento más óptimo o estable de la ejecución
            if final_f < min_f_found:
                min_f_found = final_f
                best_history_run = h
                
        all_results.append({
            'Prob': p_name, 'Pf': pf, 
            'Promedio': np.mean(runs_best), 
            'Std': np.std(runs_best), 
            'Factibles': np.sum(runs_fact)
        })
        histories_to_plot.append(best_history_run)
    
    # Despliegue de la gráfica por cada problema analizado
    graficar_convergencia(p_name, histories_to_plot, p_f_values)
    
    # NUEVA LLAMADA ADICIONAL AGREGADA AL FINAL DE LA ITERACIÓN DE CADA PROBLEMA
    graficar_comportamiento_instantaneo(p_name, histories_to_plot, p_f_values)

# IMPRESIÓN FORMAL DE LA TABLA DE RESULTADOS 
print(f"\n{'Problema':<10} | {'Pf':<5} | {'Óptimo (prom)':<15} | {'Desv. Stand':<12} | {'#Factibles'}")
print("-" * 65)
for r in all_results:
    print(f"{r['Prob']:<10} | {r['Pf']:<5} | {r['Promedio']:<15.6f} | {r['Std']:<12.6f} | {int(r['Factibles'])}")