import unittest
from diagnosis_tool import diagnose_and_recommend
import pandas as pd

class TestDiagnosisTool(unittest.TestCase):
    def test_diagnose_and_recommend(self):
        symptoms = "headache, fever"
        diagnosis, medicine = diagnose_and_recommend(symptoms)
        self.assertIsNotNone(diagnosis)
        self.assertIsNotNone(medicine)

    def test_diagnose_and_recommend_empty_symptoms(self):
        symptoms = ""
        with self.assertRaises(ValueError):
            diagnose_and_recommend(symptoms)

    def test_diagnose_and_recommend_invalid_symptoms(self):
        symptoms = " invalid symptoms "
        with self.assertRaises(ValueError):
            diagnose_and_recommend(symptoms)

    def test_diagnose_and_recommend_known_symptoms(self):
        symptoms = "headache, fever"
        diagnosis, medicine = diagnose_and_recommend(symptoms)
        self.assertEqual(diagnosis, "flu")
        self.assertEqual(medicine, "acetaminophen")

    def test_diagnose_and_recommend_unknown_symptoms(self):
        symptoms = "unknown symptoms"
        diagnosis, medicine = diagnose_and_recommend(symptoms)
        self.assertIsNotNone(diagnosis)
        self.assertIsNotNone(medicine)

if __name__ == "__main__":
    unittest.main()
