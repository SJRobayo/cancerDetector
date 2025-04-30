import joblib

# 1. Carga el modelo
model = joblib.load('model/nuevo_modelo_random_forest_optimizado.pkl')

# 2. Tipo de objeto
print("Tipo de objeto cargado:", type(model))

# 3. Si tiene feature_names_in_ (sklearn ≥1.0) muestra esas columnas
if hasattr(model, 'feature_names_in_'):
    print("\nColumnas utilizadas (feature_names_in_):")
    print(model.feature_names_in_)

# 4. Si el modelo es un Pipeline, lista sus pasos
if hasattr(model, 'named_steps'):
    print("\nPipeline steps:")
    for name, step in model.named_steps.items():
        print(f" - {name}: {step}")

# 5. Parsea los parámetros para ver hiperparámetros, etc.
print("\nParámetros del modelo (get_params):")
for k, v in model.get_params().items():
    print(f"{k}: {v}")
