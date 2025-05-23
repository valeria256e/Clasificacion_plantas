import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import shutil
import tempfile
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import numpy as np
import time
import gc

def verificar_y_filtrar_imagenes_fuerte(ruta_base):
    errores = 0
    for root, dirs, files in os.walk(ruta_base):
        for file in files:
            archivo = os.path.join(root, file)
            try:
                img = Image.open(archivo)
                img = img.convert('RGB')
                img = img.resize((224, 224))
                img.verify()
            except Exception:
                if os.path.exists(archivo):
                    os.remove(archivo)
                    print(f"Imagen corrupta eliminada: {archivo}")
                else:
                    print(f"❌ Archivo no encontrado (ya eliminado?): {archivo}")
                errores += 1
    print(f"\n✅ Total de imágenes eliminadas: {errores}")

def main():
    usar_transfer_learning = True
    num_classes = 6
    batch_size = 32
    num_epochs = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("✅ CUDA disponible:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("🖥️ GPU activa:", torch.cuda.get_device_name(0))
    else:
        print("⚠️ GPU no detectada. Estás usando CPU (muy lento)")

    # Preparar dataset
    ruta_tipoA = r"D:\Codigo\DATASET_PINOS\CUPRESSUS MACROCARPA"
    ruta_tipoB = r"D:\Codigo\DATASET_PINOS\CUPRESSUS SEMPERVIRENS"
    directorio_temporal = tempfile.mkdtemp()

    for carpeta_principal in [ruta_tipoA, ruta_tipoB]:
        nombre_tipo = os.path.basename(carpeta_principal).replace(' ', '_')
        for clase in os.listdir(carpeta_principal):
            carpeta_clase = os.path.join(carpeta_principal, clase)
            if os.path.isdir(carpeta_clase):
                nombre_clase = clase.replace(' ', '_')
                nuevo_nombre = f"{nombre_tipo}_{nombre_clase}"
                destino = os.path.join(directorio_temporal, nuevo_nombre)

                if not os.path.exists(destino):
                    shutil.copytree(carpeta_clase, destino)
                else:
                    for archivo in os.listdir(carpeta_clase):
                        ruta_origen = os.path.join(carpeta_clase, archivo)
                        ruta_destino = os.path.join(destino, archivo)
                        if not os.path.exists(ruta_destino):
                            shutil.copy2(ruta_origen, ruta_destino)

    print("✅ Imágenes copiadas")

    # verificar_y_filtrar_imagenes_fuerte(directorio_temporal)
    # gc.collect()
    # print("✅ Verificación completa")

    # Crear DataLoaders
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(directorio_temporal, transform=transform)
    class_names = dataset.classes
    print("✅ Dataset cargado. Clases detectadas:", class_names)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    print("✅ DataLoaders listos. Iniciando entrenamiento...")

    # Definir modelo
    if usar_transfer_learning:
        from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
        weights = MobileNet_V2_Weights.DEFAULT
        model = mobilenet_v2(weights=weights)
        for param in model.features.parameters():
            param.requires_grad = False
        model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    else:
        model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    model = model.to(device)
    print("🔍 Modelo está en:", next(model.parameters()).device)
    torch.backends.cudnn.benchmark = True

    # Entrenamiento
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses, val_losses = [], []
    start_time = time.time()

    for epoch in range(num_epochs):
        print(f"⏱️ Epoch {epoch+1} iniciando...")
        epoch_start = time.time()
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            print("🌀 Batch cargado")
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += torch.sum(preds == labels.data)

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        accuracy = correct.double() / len(val_dataset)

        epoch_time = time.time() - epoch_start
        print(f"✅ Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {accuracy:.4f} | Duración: {epoch_time:.2f} seg")

    print(f"\n🕒 Tiempo total de entrenamiento: {(time.time() - start_time)/60:.2f} minutos")

    # Evaluación
    print("\n📊 Evaluando el modelo con datos de validación...")
    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    accuracy_final = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    reporte_completo = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)

    print("\n✅ Resultados de evaluación:")
    print("Accuracy:", accuracy_final)
    print("Precision (macro):", precision_macro)
    print("Recall (macro):", recall_macro)
    print("F1-score (macro):", f1_macro)
    print("\nReporte por clase:\n", reporte_completo)

    matriz = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(matriz, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Matriz de Confusión')
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.tight_layout()

    # Guardar resultados
    ruta_resultados = r"D:\Resultados_CNN"
    os.makedirs(ruta_resultados, exist_ok=True)

    modelo_path = os.path.join(ruta_resultados, "modelo_plantas.pth")
    torch.save(model.state_dict(), modelo_path)
    print(f"✅ Modelo guardado en: {modelo_path}")

    grafica_path = os.path.join(ruta_resultados, "curva_perdida.png")
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Curva de pérdida')
    plt.legend()
    plt.savefig(grafica_path)
    plt.close()
    print(f"📊 Gráfica guardada en: {grafica_path}")

    conf_matrix_path = os.path.join(ruta_resultados, "matriz_confusion.png")
    plt.savefig(conf_matrix_path)
    plt.close()
    print(f"🧾 Matriz de confusión guardada en: {conf_matrix_path}")

    log_path = os.path.join(ruta_resultados, "resultados_entrenamiento.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("======= MÉTRICAS POR ÉPOCA =======\n")
        for i in range(num_epochs):
            f.write(f"Época {i+1}: Train Loss = {train_losses[i]:.4f}, Val Loss = {val_losses[i]:.4f}\n")
        f.write("\n======= MÉTRICAS FINALES =======\n")
        f.write(f"Accuracy final: {accuracy_final:.4f}\n")
        f.write(f"Precision macro: {precision_macro:.4f}\n")
        f.write(f"Recall macro: {recall_macro:.4f}\n")
        f.write(f"F1-score macro: {f1_macro:.4f}\n")
        f.write("\n======= REPORTE POR CLASE =======\n")
        f.write(reporte_completo)

    print(f"📄 Resultados escritos en: {log_path}")

    shutil.rmtree(directorio_temporal)
    print("\n🧹 Directorio temporal eliminado.")

if __name__ == "__main__":
    main()
