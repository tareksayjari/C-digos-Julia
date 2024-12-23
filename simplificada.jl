using LinearAlgebra
using Statistics
using Plots

# Calcular EMSE e RMSE
function calculate_emse_rmse(true_values, estimated_values)
    errors = true_values .- estimated_values
    emse = mean(errors.^2, dims=1)
    rmse = sqrt.(emse)
    return emse, rmse
end

# Filtro de Kalman
function kalman_filter(F, H, Q, R, P0, x0, z)
    n = length(z)
    m = length(x0)
    x = zeros(m, n)
    P = zeros(m, m, n)
    
    x[:, 1] = x0
    P[:, :, 1] = P0

    for k in 2:n
        # Prediction
        x_pred = F * x[:, k-1]
        P_pred = F * P[:, :, k-1] * F' + Q
        for i in 1:0.4*N
            D = 0.0
            for j in 1:m
                D += P_pred[j, j] * rand()
            end
        end
        # Update
        y_measurement = z[k] .- (H[:, :, k] * x_pred)[1]
        S = H[:, :, k] * P_pred * H[:, :, k]' .+ R
        K = P_pred * H[:, :, k]' * inv(S)
        for i in 1:0.4*N
            D = 0.0
            for j in 1:m
                D += K[j] * y_measurement * rand()
            end
        end

        x[:, k] = x_pred + K * y_measurement
        P[:, :, k] = P_pred - K * H[:, :, k] * P_pred
    end
    
    return x, P
end

# Equação recursiva 20
function recursive_equation_20(T, P, σ2, r, h)
    T_inv = cholesky(T).L
    n = size(P, 1)
    P_inv = zeros(n, n)
    
    for i in 1:n
        for j in 1:n
            if i == j
                P_inv[i, j] = T_inv[i, j] - T_inv[i, j] * P[i, j] * T_inv[i, j] + 1 / (σ2 * r) * h[i] * h[j]
            else
                P_inv[i, j] = -T_inv[i, i] * P[i, j] * T_inv[j, j] + 1 / (σ2 * r) * h[i] * h[j]
            end
        end
    end
    
    return P_inv
end

# Gerar valores de H
function generate_realistic_H(scenario::String, N::Int)
    H = zeros(1, 7, N)
    
    if scenario == "Sonar passivo"
        for k in 1:N
            H[:, :, k] .= reshape(1 .+ 0.1 * randn(7), 1, 7)
        end
    
    elseif scenario == "UAV em ambientes urbanos"
        for k in 1:N
            H[:, :, k] .= reshape(1 .+ 0.1 * randn(7), 1, 7)
        end
    
    elseif scenario == "Sinais PPG em dispositivos vestíveis"
        for k in 1:N
            H[:, :, k] .= reshape(1 .+ 0.1 * randn(7), 1, 7)
        end
    
    end
    
    return H
end

# Gerar dados de rastreamento de objetos subaquáticos
function generate_sonar_data(N)
    F = 1 * I(7)          
    H = generate_realistic_H("Sonar passivo", N)
    Q = 0.0001 * I(7)
    R = 0.001
    x0 = rand(7)
    z = zeros(N)
    z_true = zeros(N)
    
    x = F * x0 .+ sqrt(Q) * randn(7, N)
    for k in 1:N
        if k == 1
            x[:,k] = F * x0 .+ sqrt(Q) * randn(7, 1)
        else
            x[:,k] = F*x[:,k-1] .+ sqrt(Q) * randn(7, 1)
        end
        z_true[k] = (H[:, :, k] * x[:, k])[1]
        z[k] = z_true[k] .+ sqrt(R) * randn()
    end
    
    return F, H, Q, R, x0, z, x, z_true
end

# Gerar dados de navegação de UAV em ambientes urbanos
function generate_uav_data(N)
    F = 1 * I(7)
    H = generate_realistic_H("UAV em ambientes urbanos", N)
    Q = 0.0001 * I(7)
    R = 0.002
    x0 = rand(7)
    z = zeros(N)
    z_true = zeros(N)

    x = F * x0 .+ sqrt(Q) * randn(7, N)
    for k in 1:N
        if k == 1
            x[:,k] = F * x0 .+ sqrt(Q) * randn(7, 1)
        else
            x[:,k] = F*x[:,k-1] .+ sqrt(Q) * randn(7, 1)
        end
        z_true[k] = (H[:, :, k] * x[:, k])[1]
        z[k] = z_true[k] .+ sqrt(R) * randn()
    end
    
    return F, H, Q, R, x0, z, x, z_true
end

# Gerar dados de estimativa de frequência cardíaca a partir de sinais PPG
function generate_ppg_data(N)
    F = 1 * I(7)
    H = generate_realistic_H("Sinais PPG em dispositivos vestíveis", N)
    Q = 0.0001 * I(7)
    R = 0.003
    x0 = rand(7)
    z = zeros(N)
    z_true = zeros(N)
    
    x = F * x0 .+ sqrt(Q) * randn(7, N)
    for k in 1:N
        if k == 1
            x[:,k] = F * x0 .+ sqrt(Q) * randn(7, 1)
        else
            x[:,k] = F*x[:,k-1] .+ sqrt(Q) * randn(7, 1)
        end
        z_true[k] = (H[:, :, k] * x[:, k])[1]
        z[k] = z_true[k] .+ sqrt(R) * randn()
    end
    
    return F, H, Q, R, x0, z, x, z_true
end

# Executar a comparação dos filtros em um cenário dado
function run_experiment(F, H, Q, R, x0, z, z_true)
    P0 = I(7)
    T = 0.995 * I(7)
    σ2 = 0.01
    r = 1.0
    #################################
    x_recursive_20 = zeros(7, length(z))
    x_recursive_20[:, 1] = x0
    P_inv_20 = recursive_equation_20(T, P0, σ2, r, H[:, :, 1])
    P = inv(P_inv_20)
    #################################

    # Medir o tempo de execução e o erro do filtro de Kalman
    time_kalman = @elapsed x_kalman, P_kalman = kalman_filter(F, H, Q, R, P0, x0, z)
    z_kalman_est = [(H[:, :, k] * x_kalman[:, k])[1] for k in 1:length(z)]
    mse_kalman = mean((z_true .- z_kalman_est).^2)
    emse_kalman, rmse_kalman = calculate_emse_rmse(z_true, z_kalman_est)

    # Medir o tempo de execução e o erro da Equação recursiva 20
    P_inv_20 = zeros(7, 7)
    time_recursive_20 = @elapsed begin
        for k in 2:length(z)
            v = z[k] .- (H[:, :, k] * x_recursive_20[:, k-1])[1]
            P_inv_20 .= recursive_equation_20(T, P, σ2, r, H[:, :, k])
            α = P_inv_20 \ (H[:, :, k]' * v / σ2)
            x_recursive_20[:, k] = F * x_recursive_20[:, k-1] + F * α
            P = inv(P_inv_20)
        end
    end
    
    z_recursive_20_est = [(H[:, :, k] * x_recursive_20[:, k])[1] for k in 1:length(z)]
    mse_recursive_20 = mean((z_true .- z_recursive_20_est).^2)
    emse_recursive_20, rmse_recursive_20 = calculate_emse_rmse(z_true, z_recursive_20_est)

    return mse_kalman, mse_recursive_20, time_kalman, time_recursive_20, x_kalman, x_recursive_20, z_kalman_est, z_recursive_20_est, z_true, emse_kalman, rmse_kalman, emse_recursive_20, rmse_recursive_20
end

# Número de amostras
N = 1000

# Gerar dados para cada aplicação e executar o experimento
applications = ["Sonar passivo", "UAV em ambientes urbanos", "Sinais PPG em dispositivos vestíveis"]
results = []

for app in applications
    if app == "Sonar passivo"
        F, H, Q, R, x0, z, x, z_true = generate_sonar_data(N)
    elseif app == "UAV em ambientes urbanos"
        F, H, Q, R, x0, z, x, z_true = generate_uav_data(N)
    elseif app == "Sinais PPG em dispositivos vestíveis"
        F, H, Q, R, x0, z, x, z_true = generate_ppg_data(N)
    end
    
    mse_kalman, mse_recursive_20, time_kalman, time_recursive_20, x_kalman, x_recursive_20, z_kalman_est, z_recursive_20_est, z_true, emse_kalman, rmse_kalman, emse_recursive_20, rmse_recursive_20 = run_experiment(F, H, Q, R, x0, z, z_true)
    push!(results, (app, mse_kalman, mse_recursive_20, time_kalman, time_recursive_20, x, x_kalman, x_recursive_20, z, z_kalman_est, z_recursive_20_est, z_true, emse_kalman, rmse_kalman, emse_recursive_20, rmse_recursive_20))
end

# Mostrar os resultados
for (app, mse_kalman, mse_recursive_20, time_kalman, time_recursive_20, x, x_kalman, x_recursive_20, z, z_kalman_est, z_recursive_20_est, z_true, emse_kalman, rmse_kalman, emse_recursive_20, rmse_recursive_20) in results
    println("\nAplicação: $app")
    println("Filtro de Kalman tradicional:")
    println("Tempo de execução: ", time_kalman)
    println("MSE: ", mse_kalman)
    println("EMSE: ", emse_kalman)
    println("RMSE: ", rmse_kalman)
    println("Equação recursiva 20:")
    println("Tempo de execução: ", time_recursive_20)
    println("MSE: ", mse_recursive_20)
    println("EMSE: ", emse_recursive_20)
    println("RMSE: ", rmse_recursive_20)
end

# Separar resultados para plotagem
mse_kalman_values = []
mse_recursive_20_values = []
time_kalman_values = []
time_recursive_20_values = []

for (app, mse_kalman, mse_recursive_20, time_kalman, time_recursive_20, x, x_kalman, x_recursive_20, z, z_kalman_est, z_recursive_20_est, z_true, emse_kalman, rmse_kalman, emse_recursive_20, rmse_recursive_20) in results
    push!(mse_kalman_values, mse_kalman)
    push!(mse_recursive_20_values, mse_recursive_20)
    push!(time_kalman_values, time_kalman)
    push!(time_recursive_20_values, time_recursive_20)
end

# Configuração dos subplots
plot_layout = @layout [a b; c d; e f]

p1 = bar(["Kalman", "Recursiva 20"], [mse_kalman_values[1], mse_recursive_20_values[1]], ylabel="MSE", title="Sonar passivo")
savefig(p1, "p1_sonar_passivo_mse.png")

p2 = bar(["Kalman", "Recursiva 20"], [time_kalman_values[1], time_recursive_20_values[1]], ylabel="Tempo de execução (s)", yscale=:log10, ylim=(1e-3, 1e-2), title="Sonar passivo")
savefig(p2, "p2_sonar_passivo_tempo_execucao.png")

p3 = bar(["Kalman", "Recursiva 20"], [mse_kalman_values[2], mse_recursive_20_values[2]], ylabel="MSE", title="UAV em ambientes urbanos")
savefig(p3, "p3_uav_urbanos_mse.png")

p4 = bar(["Kalman", "Recursiva 20"], [time_kalman_values[2], time_recursive_20_values[2]], ylabel="Tempo de execução (s)", yscale=:log10, ylim=(1e-3, 1e-2), title="UAV em ambientes urbanos")
savefig(p4, "p4_uav_urbanos_tempo_execucao.png")

p5 = bar(["Kalman", "Recursiva 20"], [mse_kalman_values[3], mse_recursive_20_values[3]], ylabel="MSE", title="Sinais PPG em dispositivos vestíveis")
savefig(p5, "p5_ppg_sinais_mse.png")

p6 = bar(["Kalman", "Recursiva 20"], [time_kalman_values[3], time_recursive_20_values[3]], ylabel="Tempo de execução (s)", yscale=:log10, ylim=(1e-3, 1e-2), title="Sinais PPG em dispositivos vestíveis")
savefig(p6, "p6_ppg_sinais_tempo_execucao.png")

# Plotagem dos gráficos no layout
plot(p1, p2, p3, p4, p5, p6, layout=plot_layout)

# Plotagem dos gráficos adicionais
function summarise_x(x)
    return sum(x, dims = 1)'
end

for (app, mse_kalman, mse_recursive_20, time_kalman, time_recursive_20, x, x_kalman, x_recursive_20, z, z_kalman_est, z_recursive_20_est, z_true, emse_kalman, rmse_kalman, emse_recursive_20, rmse_recursive_20) in results
    plot1 = plot(1:N, summarise_x(x), label="x Real")
    plot!(plot1, 1:N, summarise_x(x_kalman), label="x Kalman")
    plot!(plot1, 1:N, summarise_x(x_recursive_20), label="x Recursiva 20")
    title!(plot1, "Comparação de x para $app")
    xlabel!(plot1, "Tempo")
    ylabel!(plot1, "x")
    savefig(plot1, "x_comparison_$app.png")

    plot2 = plot(1:N, z_true, label="z Real")
    plot!(plot2, 1:N, z_kalman_est, label="z Kalman")
    plot!(plot2, 1:N, z_recursive_20_est, label="z Recursiva 20")
    title!(plot2, "Comparação de z para $app")
    xlabel!(plot2, "Tempo")
    ylabel!(plot2, "z")
    savefig(plot2, "z_comparison_$app.png")
end







####################Mudanças############################
#Evitar a inversão de matrizes
#Regularization of the matrix inversion
####################Mudanças############################