# Installeren

Je hebt **Python 3.12** en **[uv](https://docs.astral.sh/uv/)** nodig. In drie commando's ben je klaar:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # 1. uv (eenmalig, macOS/Linux)
uv python pin 3.12                                # 2. Python 3.12 vastzetten
uv tool install studentprognose                   # 3. de tool installeren
```

Daarna kun je meteen door naar de [Snelstart](snelstart.md).

=== "Windows"

    ```powershell
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    uv python pin 3.12
    uv tool install studentprognose
    ```

**Updaten** naar een nieuwere versie: `uv tool upgrade studentprognose`.

!!! warning "Alleen Python 3.12 wordt ondersteund"
    De pipeline wordt op één Python-versie ontwikkeld en getest. Nieuwere minors (3.13, 3.14) zijn bewust uitgesloten — `statsforecast` publiceert daarvoor nog geen Windows-wheels, wat een C-compiler-error geeft tijdens installatie. `uv` downloadt zelf de juiste versie; je hoeft niks van python.org te halen.

## Voordat je begint

Controleer je Python-versie met `python --version` (of `python3 --version`).

??? tip "Python niet gevonden of te oud?"
    Zie je `python: command not found` of een versie lager dan 3.12? `uv python pin 3.12` regelt dit meestal vanzelf. Lukt dat niet:

    - **Windows** — download via [python.org](https://www.python.org/downloads/); vink **"Add python.exe to PATH"** aan.
    - **macOS** — [python.org](https://www.python.org/downloads/) of `brew install python@3.12`.
    - **Linux** — `sudo apt install python3.12` (Ubuntu/Debian) of `sudo dnf install python3.12` (Fedora).

## Installatie

We gebruiken **uv**, een snelle Python-pakketbeheerder die virtual environments automatisch afhandelt. De drie commando's bovenaan deze pagina zijn alles wat je nodig hebt. Werk je liever met pip, of wil je aan de broncode bijdragen? Zie hieronder.

??? note "Installeren met pip in een virtual environment"
    !!! warning "Installeer niet zonder virtual environment"
        Een `pip install` buiten een virtual environment plaatst packages in je systeemomgeving en kan conflicten veroorzaken met andere projecten.

    ```bash
    mkdir mijn-prognose && cd mijn-prognose
    python -m venv .venv
    ```

    Activeer de omgeving:

    === "Windows (PowerShell)"

        ```powershell
        .venv\Scripts\Activate.ps1
        ```

    === "Windows (CMD)"

        ```cmd
        .venv\Scripts\activate.bat
        ```

    === "macOS / Linux"

        ```bash
        source .venv/bin/activate
        ```

    Je ziet nu `(.venv)` voor je prompt. Installeer daarna:

    ```bash
    pip install studentprognose        # updaten: pip install --upgrade studentprognose
    ```

    Elke nieuwe terminal vereist een nieuwe activatie vóór `studentprognose` werkt.

??? note "Bijdragen aan de broncode"
    ```bash
    git clone https://github.com/cedanl/studentprognose.git
    cd studentprognose
    uv run studentprognose --help
    ```

    `uv run` maakt automatisch een virtual environment aan op basis van `pyproject.toml`.

## Installatie mislukt?

??? failure "`studentprognose: command not found` na installatie"
    Start je terminal opnieuw op zodat het commando beschikbaar wordt. Houdt het aan, draai dan `uv tool update-shell` en open een nieuwe terminal.

??? failure "`ModuleNotFoundError: No module named 'studentprognose'`"
    Je draait Python buiten de omgeving waarin de tool is geïnstalleerd. Gebruik `uv run studentprognose ...` om de juiste omgeving te kiezen.

??? failure "`python: command not found` of versie te laag"
    Python is niet geïnstalleerd of niet in je PATH. Zie [Voordat je begint](#voordat-je-begint). Op sommige systemen heet het commando `python3`.

Andere fouten (bijv. `TerminatedWorkerError` tijdens het draaien) staan bij [Draaien & CLI → Veelvoorkomende fouten](aan-de-slag.md#veelvoorkomende-fouten).
