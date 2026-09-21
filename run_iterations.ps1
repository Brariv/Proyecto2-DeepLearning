# Plan de iteraciones para Windows (PowerShell). Equivalente a run_iterations.sh.
# Uso, desde la carpeta del proyecto con el entorno virtual activado:
#     powershell -ExecutionPolicy Bypass -File .\run_iterations.ps1
# Ctrl+C detiene la corrida actual y guarda last.pt; se reanuda con --resume (ver README).

function Run([string[]]$argsList) {
    Write-Host "`n>> python $($argsList -join ' ')" -ForegroundColor Cyan
    & python @argsList
    if ($LASTEXITCODE -ne 0) { throw "Falló: python $($argsList -join ' ')" }
}

# Iteraciones cortas de comparación (2M pasos = 8M frames cada una)
Run @("train.py", "--preset", "dqn",       "--run-name", "it1_dqn",       "--total-steps", "2000000")
Run @("train.py", "--preset", "ddqn",      "--run-name", "it2_ddqn",      "--total-steps", "2000000")
Run @("train.py", "--preset", "dueling",   "--run-name", "it3_dueling",   "--total-steps", "2000000")
Run @("train.py", "--preset", "per_nstep", "--run-name", "it4_per_nstep", "--total-steps", "2000000")
Run @("train.py", "--preset", "rainbow",   "--run-name", "it5_rainbow",   "--total-steps", "2000000")

# Candidato final: Rainbow eficiente (IMPALA CNN). Cuantos más pasos, mejor.
Run @("train.py", "--preset", "rainbow_fast", "--run-name", "it6_rainbow_fast", "--total-steps", "10000000")

# Gráficas y tabla de todas las iteraciones
Run @("plot_results.py", "runs/it*", "--out", "figs")

# Elegir el mejor checkpoint del candidato final (10 episodios por checkpoint)
Run @("evaluate.py", "--checkpoints-dir", "runs/it6_rainbow_fast", "--episodes", "10", "--json", "figs/ranking_ckpts.json")

# Competencia: 5 episodios greedy + video
Run @("evaluate.py", "--checkpoint", "runs/it6_rainbow_fast/best.pt", "--episodes", "5", "--video", "videos/agente_final.mp4")
