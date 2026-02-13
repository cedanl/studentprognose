# Technische README

## Installatie

### Vereisten
- **Besturingssysteem**: Windows, macOS of Linux
- **Software**: [uv](https://docs.astral.sh/uv/getting-started/installation/)

### Stappen

1. **Installeer uv**  
   - **Linux/macOS**:  
     ```bash
     curl -LsSf https://astral.sh/uv/install.sh | sh
     ```
   - **Windows**:  
     ```powershell
     powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
     ```

2. **Controleer de installatie**  
   ```bash
   uv self update
   ```

3. **Start de applicatie**  
   ```bash
   uv run main.py
   ```


## Architectuur

Het model bestaat uit de volgende componenten:
- **Backend**: Verantwoordelijk voor dataverwerking en voorspellingen.
- **Frontend**: CLI-gebaseerde interface.
- **Data**: Input- en outputbestanden.

### Belangrijke bestanden
- `main.py`: Startpunt van de applicatie.
- `scripts/`: Kernlogica van het model.
- `configuration/`: Configuratiebestanden.

