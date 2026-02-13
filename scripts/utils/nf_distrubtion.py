import numpy as np

def _nf_students_based_on_distribution_of_last_years(self, prediction, examentype, programme, herkomst, predict_year, predict_week):
        last_years_data = self.data_latest[
            (self.data_latest["Collegejaar"] < predict_year)
            & (self.data_latest["Collegejaar"] >= predict_year - 3)
            & (self.data_latest["Weeknummer"] == predict_week)
            & (self.data_latest["Croho groepeernaam"] == programme)
            & (self.data_latest["Examentype"] == examentype)
        ].fillna(0)

        # Initialize a list to store distributions per year
        distributions = []

        for last_year in range(predict_year - 3, predict_year):
            year_data = last_years_data[last_years_data["Collegejaar"] == last_year]

            total_students = year_data["Aantal_studenten"].sum()
            if total_students == 0:
                continue  # Skip if no data or avoid division by zero

            herkomst_students = year_data[year_data["Herkomst"] == herkomst]["Aantal_studenten"].sum()

            distributions.append(herkomst_students / total_students)

        # Compute mean distribution across available years
        distribution = np.mean(distributions) if distributions else 0
            
        # Validate predection
        try:
            if prediction > self.configuration['numerus_fixus'][programme] * distribution:
                prediction = self.configuration['numerus_fixus'][programme] * distribution
        except KeyError:
            pass
            #if prediction > self.configuration['used_to_be_numerus_fixus'][programme] * distribution:
            #    prediction = self.configuration['used_to_be_numerus_fixus'][programme] * distribution
        
        return round(prediction)