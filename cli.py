import argparse
import datetime

def parse_args():
    """
    Parses command-line arguments using argparse for a more robust and readable CLI.

    Returns:
        argparse.Namespace: An object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Student Prognosis Model CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # --- Argument Definitions ---
    parser.add_argument(
        '-w', '--weeks',
        nargs='+',
        type=str, 
        help='One or more week numbers or ranges (e.g., 5 6 7 or 10:15 or 39:38).'
    )
    parser.add_argument(
        '-y', '--years',
        nargs='+',
        type=str,  
        help='One or more academic years or ranges (e.g., 2023 2024 or 2022:2025).'
    )
    parser.add_argument(
        '-wf', '--write-file',
        action='store_true',
        help='Write predictions to the total file.'
    )
    parser.add_argument(
        '-p', '--print',
        action='store_true',
        help='Print programme output.'
    )
    parser.add_argument(
        '-ev', '--evaluate',
        action='store_true',
        help='Evaluate the predictions.'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Prints the model output'
    )

    args = parser.parse_args()

    # --- Post-processing and Default Value Logic ---
    def parse_week_token(token):
        """Parse a single week token (supports wrap-around like 39:38)."""
        if ':' in token:
            start, end = map(int, token.split(':'))
            if start <= end:
                return list(range(start, end + 1))
            else:
                # wrap-around case: e.g., 39:38 → 39..52 + 1..38
                return list(range(start, 53)) + list(range(1, end + 1))
        else:
            return [int(token)]

    def parse_year_token(token):
        """Parse a single year token (simple linear range)."""
        if ':' in token:
            start, end = map(int, token.split(':'))
            if start <= end:
                return list(range(start, end + 1))
            else:
                # Optional: handle reversed input gracefully
                return list(range(end, start + 1))
        else:
            return [int(token)]

    # --- Process weeks ---
    if args.weeks:
        weeks = []
        for token in args.weeks:
            weeks.extend(parse_week_token(token))
        args.weeks = weeks  # preserve chronological order
    else:
        current_week = datetime.date.today().isocalendar()[1]
        args.weeks = [min(current_week, 52)]

    # --- Process years ---
    if args.years:
        years = []
        for token in args.years:
            years.extend(parse_year_token(token))
        args.years = sorted(set(years))
    else:
        # Default year logic (based on current week)
        if datetime.date.today().isocalendar()[1] >= 39:
            args.years = [datetime.date.today().year + 1]
        else:
            args.years = [datetime.date.today().year]

    return args