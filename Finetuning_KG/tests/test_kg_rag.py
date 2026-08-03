import os
import sys
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from kg_rag.context_builder import build_kg_context
from kg_rag.kg_loader import ClinicalKG


class KgRagRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kg = ClinicalKG(os.path.join(ROOT, "kg_rag", "clinical_kg.json"))

    def context(self, report):
        return build_kg_context(report, self.kg)

    def test_bilateral_values_keep_correct_sides(self):
        context = self.context(
            "横轴位左侧脑室三角区宽约1.12cm，右侧宽约1.02cm。"
        )
        self.assertIn("左侧侧脑室三角区宽度宽约1.12cm", context)
        self.assertIn("右侧侧脑室三角区宽度宽约1.02cm", context)

    def test_cervix_does_not_capture_cyst_dimensions(self):
        context = self.context(
            "宫颈管长度3.63cm，母体阴道后壁囊状影大小约1.4cm×1.2cm。"
        )
        self.assertNotIn("宫颈管长度长约1.4cm", context)
        self.assertNotIn("宫颈管长度长约1.2cm", context)

    def test_only_first_value_after_structure_anchor_is_used(self):
        context = self.context(
            "宫颈管长约3.6cm，其旁囊性灶大小约1.4cm×1.2cm。"
        )
        self.assertNotIn("宫颈管长度", context)

    def test_restored_measurement_aliases(self):
        context = self.context(
            "宫颈管长约2.2cm；透明隔间腔最宽约1.3cm；枕大池宽约1.2cm。"
        )
        self.assertIn("宫颈管长度长约2.2cm", context)
        self.assertIn("透明隔腔宽度宽约1.3cm", context)
        self.assertIn("枕大池深度深约1.2cm", context)

    def test_finding_uses_interpretation_text(self):
        context = self.context("胎儿胼胝体缺如，双侧侧脑室平行分离。")
        self.assertIn("提示胼胝体发育异常", context)
        self.assertNotIn("解释_胼胝体发育异常", context)

    def test_positive_finding_alias_containing_negation_word(self):
        context = self.context("胎儿胼胝体未显示，双侧侧脑室平行分离。")
        self.assertIn("提示胼胝体发育异常", context)

    def test_negated_finding_is_not_added(self):
        context = self.context("胼胝体显示，形态未见明确异常。")
        self.assertNotIn("提示胼胝体发育异常", context)


if __name__ == "__main__":
    unittest.main()
