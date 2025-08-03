using Random
using LinearAlgebra
using DataFrames
using StatsBase
using Plots
using StatsPlots
using DataFrames
using StatsPlots  # Utilisez StatsPlots pour les graphiques statistiques
using CSV  # Pour charger le fichier CSV
using Statistics  # Pour les statistiques de base comme la moyenne, l'écart type, etc.
using Correlations

# Fixer la seed pour la reproductibilité
Random.seed!(42)

# Nombre de lignes
n_rows = 10000

# Générer les caractéristiques
surface = rand(n_rows) * (300 - 30) .+ 30  # Surface entre 30 et 300 m²
age = rand(0:49, n_rows)  # Âge entre 0 et 49 ans
security = rand(1:5, n_rows)  # Sécurité entre 1 et 5
localization = rand(1:5, n_rows)  # Localisation entre 1 et 5
num_bedrooms = rand(1:5, n_rows)  # Chambres entre 1 et 5
equipment = rand(1:5, n_rows)  # Équipement entre 1 et 5

# Définir la relation pour le prix
price = 2000 .* surface .- 500 .* age .+ 1500 .* security .+ 3000 .* localization .+ 10000 .* num_bedrooms .+ 2000 .* equipment .+ randn(n_rows) * 10000

# Créer le DataFrame
data = DataFrame(Surface=surface, Age=age, Security=security, Localization=localization, Bedrooms=num_bedrooms, Equipment=equipment, Price=price)

# Ajouter des outliers
n_outliers = round(Int, 0.02 * n_rows)  # 2% des données comme outliers
outlier_indices = rand(1:n_rows, n_outliers)

data[outlier_indices, :Surface] .= data[outlier_indices, :Surface] .* rand(n_outliers) .* 2 .+ 2
data[outlier_indices, :Price] .= data[outlier_indices, :Price] .* rand(n_outliers) .* 2 .+ 2
data[outlier_indices, :Age] .= round.(Int, data[outlier_indices, :Age] .* rand(n_outliers) .* 1.5 .+ 1.5)  # Ensure integer values for Age

# Préparer X et y
X = hcat(ones(n_rows), Matrix(data[:, 1:6]))  # Ajouter une colonne de biais à X
y = data[:, :Price]

# Normalisation des données
train_mean = mean(X[:, 2:end], dims=1)
train_std = std(X[:, 2:end], dims=1)
X_normalized = (X[:, 2:end] .- train_mean) ./ train_std
X = hcat(ones(n_rows), X_normalized)

# Division des données en training (80%) et testing (20%)
n = size(X, 1)
train_indices = randperm(n)[1:round(Int, 0.8 * n)]
test_indices = setdiff(1:n, train_indices)

X_train = X[train_indices, :]
y_train = y[train_indices]
X_test = X[test_indices, :]
y_test = y[test_indices]

# Vérification du rang de X_train pour détecter les problèmes d'inversibilité
println("Rang de X_train : ", rank(X_train))

# Vérifier si la matrice est inversible avant de procéder à l'inversion
if rank(X_train) == size(X_train, 2)
    # Calcul de l'équation normale avec régularisation Ridge
    λ = 1e-4  # Paramètre de régularisation
    I = λ .* LinearAlgebra.I(size(X_train, 2))  # Matrice identité avec régularisation
    theta_normal = (X_train' * X_train + I) \ (X_train' * y_train)
else
    error("La matrice X_train n'est pas inversible.")
end

# Définir la fonction de descente de gradient avec des arguments nommés
function gradient_descent(X, y; learning_rate=0.001, epochs=10000, batch_size=128, tolerance=1e-6)
    n = size(X, 1)
    theta = zeros(size(X, 2))
    prev_cost = Inf  # Initialisation de l'erreur précédente
    
    for epoch in 1:epochs
        indices = randperm(n)
        X = X[indices, :]
        y = y[indices]
        
        for i in 1:batch_size:n
            end_idx = min(i + batch_size - 1, n)
            X_batch = X[i:end_idx, :]
            y_batch = y[i:end_idx]
            
            predictions = X_batch * theta
            errors = predictions .- y_batch
            gradients = X_batch' * errors / length(y_batch)
            
            theta .-= learning_rate * gradients
        end
        
        # Calculer la fonction de coût (MSE) pour "early stopping"
        cost = mean((X * theta .- y) .^ 2)
        if abs(prev_cost - cost) < tolerance
            println("Convergence atteinte après $epoch époques")
            break
        end
        prev_cost = cost
    end
    
    return theta
end

# Entraînement avec descente de gradient sur les données de training
theta_gd = gradient_descent(X_train, y_train; learning_rate=0.001, epochs=5000, batch_size=128)

# Prédiction avec les deux méthodes
y_pred_gd = X_test * theta_gd
y_pred_normal = X_test * theta_normal

# Calcul des erreurs MSE (Mean Squared Error)
mse_gd = mean((y_test .- y_pred_gd) .^ 2)
mse_normal = mean((y_test .- y_pred_normal) .^ 2)

# Calcul du R² (coefficient de détermination)
r_squared_gd = 1 - sum((y_test .- y_pred_gd) .^ 2) / sum((y_test .- mean(y_test)) .^ 2)
r_squared_normal = 1 - sum((y_test .- y_pred_normal) .^ 2) / sum((y_test .- mean(y_test)) .^ 2)

# Calcul de la précision (1 - MSE / variance)
precision_gd = 1 - mse_gd / var(y_test)
precision_normal = 1 - mse_normal / var(y_test)

# Affichage des résultats
println("MSE avec la descente de gradient (mini-batch) : $mse_gd")
println("MSE avec l'équation normale : $mse_normal")
println("R² avec la descente de gradient (mini-batch) : $r_squared_gd")
println("R² avec l'équation normale : $r_squared_normal")
println("Précision avec la descente de gradient (mini-batch) : $precision_gd")
println("Précision avec l'équation normale : $precision_normal")

# Afficher les résultats des coefficients pour comparer
println("Coefficients obtenus par la descente de gradient :")
println(theta_gd)

println("Coefficients obtenus par l'équation normale :")
println(theta_normal)

#analyse univariee

# Charger votre dataset (remplacez le chemin par le vôtre)
my_data = data

# Choisir un backend pour Plots
gr()  # Utilisez le backend GR (vous pouvez aussi essayer plotly() si vous préférez)

# Boucle pour chaque colonne
for col in names(my_data)
    println("Analyse de la variable :", col)

    # Extraction de la colonne
    column = my_data[!, col]

    # Vérification du type de colonne
    if eltype(column) <: Union{Float64, Int64}
        # Statistiques descriptives
        println("Résumé statistique :")
        clean_column = skipmissing(column)  # Enlever les valeurs manquantes
        println("Moyenne: ", mean(clean_column))
        println("Écart-type: ", std(clean_column))
        println("Min: ", minimum(clean_column))
        println("Max: ", maximum(clean_column))

        # Histogramme
        p1 = histogram(column, bins=10, color=:blue, alpha=0.7, title="Histogramme de $col",
                       xlabel=col, ylabel="Fréquence", legend=false)
        density!(column, color=:red, lw=2, label="Densité")  # Utilisation de density! de StatsPlots
        display(p1)  # Afficher le graphique
        savefig("histogramme_$col.png")  # Sauvegarder le graphique

        # Boîte à moustaches
        p2 = boxplot(column, color=:cyan, alpha=0.7, title="Boîte à moustaches de $col",
                     ylabel=col, legend=false)
        display(p2)  # Afficher le graphique
        savefig("boxplot_$col.png")  # Sauvegarder le graphique

        # Calcul de l'asymétrie
        skew = sum((column .- mean(clean_column)).^3) /
               ((length(column) - 1) * std(clean_column)^3)

        if skew > 0.5
            println("La distribution de $col est positivement asymétrique (skew > 0.5).")
        elseif skew < -0.5
            println("La distribution de $col est négativement asymétrique (skew < -0.5).")
        else
            println("La distribution de $col est relativement symétrique.")
        end

    elseif eltype(column) <: String
        # Diagramme en barres
        counts = countmap(column)
        p3 = bar(keys(counts), values(counts), color=:orange, alpha=0.7,
                 title="Diagramme en barres de $col", xlabel=col, ylabel="Fréquence",
                 xticks=keys(counts), rotation=45, legend=false)
        display(p3)  # Afficher le graphique
        savefig("barplot_$col.png")  # Sauvegarder le graphique

        # Valeurs les plus fréquentes
        println("La variable $col est catégorielle. Les valeurs les plus fréquentes sont :")
        println(sort(counts, by=x->x[2], rev=true)[1:5])
    else
        println("Type de données inconnu pour la colonne $col, impossible de générer un graphique.")
    end
    println(repeat("-", 30))  # Séparation entre les analyses
end


# analyse bivariée

# Fonction pour l'analyse bivariée
function analyse_bivariee(data::DataFrame)
    variables = names(data)
    
    # Pour chaque paire de variables
    for i in 1:length(variables)
        for j in (i+1):length(variables)
            var1 = variables[i]
            var2 = variables[j]
            
            println("\nAnalyse de la relation entre $var1 et $var2")
            println(repeat("-", 50))
            
            col1 = data[!, var1]
            col2 = data[!, var2]
            
            # Vérification des types de colonnes
            if (eltype(col1) <: Union{Float64, Int64}) && (eltype(col2) <: Union{Float64, Int64})
                # Analyse pour variables numériques
                
                # Calcul de la corrélation
                clean_data = collect(zip(skipmissing(col1), skipmissing(col2)))
                if length(clean_data) > 0
                    x = [p[1] for p in clean_data]
                    y = [p[2] for p in clean_data]
                    correlation = cor(x, y)
                    println("Coefficient de corrélation: ", round(correlation, digits=3))
                    
                    # Nuage de points
                    p1 = scatter(x, y,
                        title="Nuage de points: $var1 vs $var2",
                        xlabel=var1,
                        ylabel=var2,
                        alpha=0.6,
                        legend=false
                    )
                    # Ajouter la ligne de régression
                    if !isnan(correlation)
                        fit = [ones(length(x)) x] \ y
                        plot!(x, fit[1] .+ fit[2].*x, color=:red, lw=2)
                    end
                    display(p1)
                    savefig("scatter_$(var1)_$(var2).png")
                    
                    # Heatmap de densité
                    p2 = histogram2d(x, y,
                        title="Heatmap de densité: $var1 vs $var2",
                        xlabel=var1,
                        ylabel=var2,
                        color=:viridis
                    )
                    display(p2)
                    savefig("heatmap_$(var1)_$(var2).png")
                    
                end
                
            elseif (eltype(col1) <: String) && (eltype(col2) <: Union{Float64, Int64})
                # Analyse pour variable catégorielle vs numérique
                
                # Boîtes à moustaches par catégorie
                p3 = boxplot(col1, col2,
                    title="Distribution de $var2 par $var1",
                    xlabel=var1,
                    ylabel=var2,
                    rotation=45
                )
                display(p3)
                savefig("boxplot_$(var1)_$(var2).png")
                
                # Statistiques par groupe
                println("\nStatistiques par groupe:")
                for groupe in unique(col1)
                    groupe_data = col2[col1 .== groupe]
                    println("$groupe:")
                    println("  Moyenne: ", round(mean(skipmissing(groupe_data)), digits=2))
                    println("  Écart-type: ", round(std(skipmissing(groupe_data)), digits=2))
                    println("  N: ", count(!ismissing, groupe_data))
                end
                
            elseif (eltype(col1) <: String) && (eltype(col2) <: String)
                # Analyse pour variables catégorielles
                
                # Table de contingence
                contingency = DataFrame(zeros(Int, length(unique(col1)), length(unique(col2))),
                                     Symbol.(unique(col2)))
                rownames = unique(col1)
                
                for (i, val1) in enumerate(unique(col1))
                    for (j, val2) in enumerate(unique(col2))
                        contingency[i,j] = sum((col1 .== val1) .& (col2 .== val2))
                    end
                end
                
                println("\nTable de contingence:")
                println(contingency)
                
                # Test du chi-carré
                observed = Matrix(contingency)
                chi = ChisqTest(observed)
                println("\nTest du chi-carré:")
                println("p-valeur: ", round(pvalue(chi), digits=4))
                
                # Graphique en mosaïque
                p4 = groupedbar(observed,
                    title="Distribution de $var2 par $var1",
                    xlabel=var1,
                    ylabel="Fréquence",
                    label=reshape(unique(col2), 1, :),
                    rotation=45
                )
                display(p4)
                savefig("mosaic_$(var1)_$(var2).png")
            end
        end
    end
end



# Exemple d'utilisation

resultats = analyse_bivariee(data)
println(resultats)