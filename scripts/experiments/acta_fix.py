import os

# Read the file
with open('aprendizaje.md', 'r', encoding='utf-8', errors='replace') as f:
    content = f.read()

# Retoque 1: Cambiar "Causa raíz: Bug en la construcción/evaluación del lote (afecta a random, z-score y candidato)"
# por: "Causa sin localizar; candidatas: construcción del brazo random, alineación scores-etiquetas, función de AUC del lote. Solo el random en 0.010 es imposible por sí mismo."
old1 = "Causa raíz: Bug en la construcción/evaluación del lote (afecta a random, z-score y candidato)"
new1 = "Causa sin localizar; candidatas: construcción del brazo random, alineación scores-etiquetas, función de AUC del lote. Solo el random en 0.010 es imposible por sí mismo."
content = content.replace(old1, new1)

# Retoque 2: Cambiar "El bug afecta a los tres brazos" por "posible causa común"
old2 = "El bug afecta a los tres brazos"
new2 = "posible causa común"
content = content.replace(old2, new2)

# Retoque 3: Cambiar el cierre sobre roles
# Quitar la línea que dice el user debe ejecutar y poner que soy yo
old3_part1 = "Cuándo la tengas, pásame los números"
new3_part1 = "Cuando ejecutes el protocolo de 10 minutos, pasame los números y yo te diré cuál de las tres ramas se cumple (G1 random, G2 z-score o G3 cobertura)"

# Buscar y reemplazar esta línea específica
if old3_part1 in content:
    content = content.replace(old3_part1, new3_part1)
else:
    # Buscar variaciones
    for line in content.split('\n'):
        if 'Cuándo la tengas' in line:
            idx = content.index(line)
            content = content.replace(line, new3_part1)
            break

# También buscar y reemplazar la inversa - que dice "decidirás cuando"
old3_part2 = "decidirás cuando yo ejecute"
new3_part2 = "Yo ejecuto el protocolo y te paso los resultados, luego decides el veredicto"
content = content.replace(old3_part2, new3_part2)

# Escribir de vuelta
with open('aprendizaje.md', 'w', encoding='utf-8') as f:
    f.write(content)

print("Retoques aplicados exitosamente")