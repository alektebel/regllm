#!/bin/bash
# Script de configuración para el proyecto de análisis bancario

echo "==================================="
echo "CONFIGURACIÓN - ANÁLISIS BANCARIO"
echo "==================================="

# Verificar Python
if ! command -v python3 &> /dev/null; then
    echo "✗ Error: Python 3 no está instalado"
    exit 1
fi

echo "✓ Python encontrado: $(python3 --version)"

# Instalar dependencias Python
echo ""
echo "[1/2] Instalando dependencias Python..."
pip3 install -r requirements-banking.txt

# Verificar poppler-utils (para pdftotext)
echo ""
echo "[2/2] Verificando poppler-utils (pdftotext)..."
if ! command -v pdftotext &> /dev/null; then
    echo "⚠️  pdftotext no encontrado. Instalando..."
    
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        if command -v apt-get &> /dev/null; then
            sudo apt-get update && sudo apt-get install -y poppler-utils
        elif command -v dnf &> /dev/null; then
            sudo dnf install -y poppler-utils
        elif command -v pacman &> /dev/null; then
            sudo pacman -S poppler
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        brew install poppler
    fi
else
    echo "✓ pdftotext encontrado"
fi

# Crear directorios si no existen
mkdir -p data/raw data/processed data/finetuning

echo ""
echo "==================================="
echo "✓ CONFIGURACIÓN COMPLETADA"
echo "==================================="
echo ""
echo "Siguiente paso: Ejecutar el script de descarga"
echo "  python3 scripts/download_financial_reports.py"
