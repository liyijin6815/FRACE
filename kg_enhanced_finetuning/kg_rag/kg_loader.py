"""知识图谱加载与索引模块"""
import json
from collections import defaultdict


class ClinicalKG:
    """临床知识图谱加载器，构建索引以支持高效查询"""

    def __init__(self, kg_path):
        """加载知识图谱并构建索引

        Args:
            kg_path: 知识图谱JSON文件路径
        """
        with open(kg_path, encoding="utf-8") as f:
            data = json.load(f)

        self.entities = data["entities"]
        self.triples = data["triples"]
        self._build_index()

    def _build_index(self):
        """构建索引结构以支持快速查询"""
        # 1) 邻接表：spo[s][p] = [triple, ...]，支持沿关系遍历
        self.spo = defaultdict(lambda: defaultdict(list))
        for t in self.triples:
            self.spo[t["s"]][t["p"]].append(t)

        # 2) 别名表：alias -> 标准实体名（用于概念匹配）
        self.alias2entity = {}
        for name, attr in self.entities.items():
            self.alias2entity[name] = name
            for alias in attr.get("aliases", []):
                self.alias2entity[alias] = name

    def get_entity(self, name):
        """获取实体属性

        Args:
            name: 实体名称

        Returns:
            实体属性字典，不存在则返回None
        """
        return self.entities.get(name)

    def get_objects(self, s, p):
        """获取所有满足 (s, p, ?) 的尾实体o

        Args:
            s: 头实体
            p: 关系谓词

        Returns:
            尾实体列表
        """
        return [t["o"] for t in self.spo[s][p]]

    def get_triples(self, s, p):
        """获取所有满足 (s, p, ?) 的完整三元组（带meta）

        Args:
            s: 头实体
            p: 关系谓词

        Returns:
            三元组列表
        """
        return self.spo[s][p]

    def entities_by_type(self, etype):
        """按类型获取所有实体

        Args:
            etype: 实体类型（Measurement, Finding等）

        Returns:
            实体名称列表
        """
        return [name for name, attr in self.entities.items()
                if attr.get("type") == etype]

    def resolve_alias(self, alias):
        """将别名解析为标准实体名

        Args:
            alias: 别名或标准名

        Returns:
            标准实体名，不存在则返回None
        """
        return self.alias2entity.get(alias)
