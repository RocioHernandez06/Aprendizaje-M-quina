using Statistics
using Random, Distributions, DataFrames, StatsBase, GLM, Plots
using DecisionTree
using KernelDensity
using StatsPlots
using Random, Distributions
# Datos originales: lluvia en 3 días (mm)
lluvia = [10, 25, 40]
# Número de réplicas en cada remuestreo
n = length(lluvia)
# Generar TODOS los posibles remuestreos con reemplazo
remuestreos = [[lluvia[rand(1:n)] for _ in 1:n] for _ in 1:27]  # 3^3 = 27 casos
# Calcular la media de cada remuestreo
medias = [mean(r) for r in remuestreos]
# Mostrar los primeros remuestreos
for (i, r) in enumerate(remuestreos[1:10])  # mostrar solo 10 de ejemplo
    println("Remuestreo $i: ", r, " → media = ", mean(r))
end
println("\nDistribución de medias bootstrap:")
println(medias)
println("Media observada (original): ", mean(lluvia))
println("Media promedio de remuestreos: ", mean(medias))





Random.seed!(123)

# Simulamos 20 días de lluvia (mm)
lluvia = rand(Gamma(2, 15), 20)   # media ≈ 30 mm

# Media observada
media_obs = mean(lluvia)
println("Media de lluvia observada (20 días): ", round(media_obs, digits=2), " mm")

# Bootstrap
B = 1000
medias_boot = [mean(sample(lluvia, 20, replace=true)) for _ in 1:B]

# Error estándar bootstrap
se_boot = std(medias_boot)
println("Error estándar bootstrap: ", round(se_boot, digits=2))

# Intervalo de confianza (percentil al 95%)
ic_95 = quantile(medias_boot, [0.025, 0.975])
println("IC 95% bootstrap de la media de lluvia: ", round.(ic_95, digits=2), " mm")



# ================================
# 1. Dataset realista de inundaciones
# ================================
Random.seed!(123)
n = 300

lluvia_total      = rand(Gamma(3, 25), n)                 # media ≈ 75 mm
intensidad_lluvia = rand(Gamma(2, 15), n)                 # media ≈ 30 mm/h
duracion_lluvia   = rand(LogNormal(log(2), 0.4), n)       # h
capacidad_drenaje = rand(LogNormal(log(60), 0.3), n)      # l/s
impermeabilidad   = rand(Beta(5, 2), n)                   # tendencia alta

riesgo = 0.5 .* (lluvia_total ./ 100) .+
         0.3 .* (intensidad_lluvia ./ 70) .+
         0.2 .* impermeabilidad .-
         0.2 .* (capacidad_drenaje ./ 100) .+
         0.3 .* (duracion_lluvia ./ 6)

prob_inundacion = 1 ./ (1 .+ exp.(-5 .* (riesgo .- 0.5)))
zona_inundada = Int.(rand.(Bernoulli.(prob_inundacion)))

df = DataFrame(
    lluvia=lluvia_total,
    intensidad=intensidad_lluvia,
    duracion=duracion_lluvia,
    drenaje=capacidad_drenaje,
    impermeabilidad=impermeabilidad,
    zona_inundada=zona_inundada
)

print(df)
# ================================
# 2. Modelos base
# ================================
# Regresión logística
modelo_log = glm(@formula(zona_inundada ~ lluvia + intensidad + duracion + drenaje + impermeabilidad),
                 df, Binomial(), LogitLink())
acc_log = mean(round.(GLM.predict(modelo_log)) .== df.zona_inundada)

# Árbol de decisión
X_matrix = Matrix(df[:, Not(:zona_inundada)])
y_vector = df.zona_inundada
modelo_tree = DecisionTree.DecisionTreeClassifier(max_depth=4)
DecisionTree.fit!(modelo_tree, X_matrix, y_vector)
pred_tree = DecisionTree.predict(modelo_tree, X_matrix)
acc_tree = mean(pred_tree .== y_vector)

println("Precisión Logística: ", acc_log)
println("Precisión Árbol: ", acc_tree)

# ================================
# 3. Bootstrap de accuracy
# ================================
n_boot = 3000
accs_log = Float64[]
accs_tree = Float64[]

for b in 1:n_boot
    idxs = sample(1:n, n, replace=true)
    df_b = df[idxs, :]

    # logística
    mlog_b = glm(@formula(zona_inundada ~ lluvia + intensidad + duracion + drenaje + impermeabilidad),
                 df_b, Binomial(), LogitLink())
    push!(accs_log, mean(round.(GLM.predict(mlog_b)) .== df_b.zona_inundada))

    # árbol
    X_b = Matrix(df_b[:, Not(:zona_inundada)])
    y_b = df_b.zona_inundada
    mtree_b = DecisionTree.DecisionTreeClassifier(max_depth=4)
    DecisionTree.fit!(mtree_b, X_b, y_b)
    pred_tb = DecisionTree.predict(mtree_b, X_b)
    push!(accs_tree, mean(pred_tb .== y_b))
end

# Intervalos de confianza
ic_log = quantile(accs_log, [0.025, 0.975])
ic_tree = quantile(accs_tree, [0.025, 0.975])
println("IC Logística: ", ic_log)
println("IC Árbol: ", ic_tree)

# ================================
# 4. Visualizaciones bootstrap
# ================================
histogram(accs_log, bins=20, alpha=0.5, color=:blue,
    xlabel="Accuracy", ylabel="Frecuencia",
    title="Bootstrap Accuracy - Logística", legend=false)

histogram(accs_tree, bins=20, alpha=0.5, color=:green,
    xlabel="Accuracy", ylabel="Frecuencia",
    title="Bootstrap Accuracy - Árbol", legend=false)

# Comparación
histogram(accs_log, bins=20, alpha=0.5, color=:blue,
    label="Logística", xlabel="Accuracy", ylabel="Frecuencia",
    title="Distribución Bootstrap - Comparación")
histogram!(accs_tree, bins=20, alpha=0.5, color=:green, label="Árbol")

# Boxplot
group_labels = vcat(fill("Logística", n_boot), fill("Árbol", n_boot))
acc_values = vcat(accs_log, accs_tree)
boxplot(group_labels, acc_values, legend=false,
    ylabel="Accuracy", title="Boxplot Bootstrap Accuracy")

# KDE
density(accs_log, label="Logística", color=:blue,
    xlabel="Accuracy", ylabel="Densidad",
    title="Distribuciones Bootstrap (KDE)")
density!(accs_tree, label="Árbol", color=:green)

# Evolución accuracy
plot(1:n_boot, accs_log, label="Logística", color=:blue,
    xlabel="Iteración Bootstrap", ylabel="Accuracy",
    title="Evolución del Accuracy")
plot!(1:n_boot, accs_tree, label="Árbol", color=:green)


# =============================
# 2. Modelo con un único árbol
# =============================
tree = DecisionTree.DecisionTreeClassifier(max_depth=3)
DecisionTree.fit!(tree, X_matrix, y_vector)
y_pred = DecisionTree.predict(tree, X_matrix)
acc_single = mean(y_pred .== y_vector)

println("Accuracy de un solo árbol: ", round(acc_single, digits=3))

# =============================
# 3. Bagging manual (bootstrap)
# =============================
function bagging(X, y; B=50, max_depth=3)
    n = size(X,1)
    preds = zeros(B, n)  # guardar predicciones de cada árbol

    for b in 1:B
        idxs = sample(1:n, n, replace=true)   # remuestreo bootstrap
        Xb = X[idxs, :]
        yb = y[idxs]

        tree_b = DecisionTree.DecisionTreeClassifier(max_depth=max_depth)
        DecisionTree.fit!(tree_b, Xb, yb)

        preds[b, :] = DecisionTree.predict(tree_b, X)   # predecimos sobre todo X
    end

    # Promedio de predicciones -> mayoría
    y_final = [mean(preds[:,i]) >= 0.5 ? 1 : 0 for i in 1:n]
    acc = mean(y_final .== y)

    return acc, y_final
end

# =============================
# 4. Evaluar Bagging con distintos números de árboles
# =============================
n_trees = 1:50
accs_bagging = [bagging(X_matrix, y_vector, B=b)[1] for b in n_trees]

# =============================
# 5. 📌 Clasificación final para un número específico de árboles
# =============================
acc_30, y_final_30 = bagging(X_matrix, y_vector, B=30)

n_inundados = count(==(1), y_final_30)
n_no_inundados = count(==(0), y_final_30)

println("\nResultados Bagging con 30 árboles:")
println("Accuracy: ", round(acc_30, digits=3))
println("Inundados (1): ", n_inundados)
println("No Inundados (0): ", n_no_inundados)

# =============================
# 5. Visualización
# =============================
plot(n_trees, accs_bagging,
     xlabel="Número de árboles en Bagging",
     ylabel="Accuracy",
     title="Bagging en predicción de inundaciones",
     label="Bagging (promedio)",
     lw=2, color=:blue)

hline!([acc_single], label="Un solo árbol", color=:red, lw=2, ls=:dash)




