# tools/sample_selector_tool.py
import os
import logging
from typing import Type, Optional, Dict, Any
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr
from scripts.selector import SampleSelector

logger = logging.getLogger(__name__)

class SelectSamplesInput(BaseModel):
    design_input: Dict[str, Any] = Field(
        description="当前设计任务的工程特征字典，应包含 '线路类型'（整体式/分离式）、地形数据、构造物信息等字段。通常来自数据加载工具的输出。"
    )
    k: int = Field(
        default=3,
        description="需要返回的最相似样本数量，默认 3 个。",
        ge=1,
        le=10
    )

class SampleSelectorTool(BaseTool):
    name: str = "select_design_samples"
    description: str = """
    从历史优秀设计样本库中，根据当前设计输入的工程特征（线路类型、地形高差、构造物数量、最小平曲线半径等），
    选择最相似的 k 个样本，供后续设计生成时参考或作为 Few-Shot 示例。
    返回的是样本字典列表，每个样本包含完整的输入输出结构（平曲线、横断面、纵断面、设计布跨等）。
    """
    args_schema: Type[BaseModel] = SelectSamplesInput
    
    # 声明为 Pydantic 字段，但设为可选，默认 None，在 __init__ 中赋值
    few_shots_dir: str = Field(default="data/few_shots", description="样本库目录路径")
    
    # 用 PrivateAttr 标记不需要验证的私有属性
    _selector: SampleSelector = PrivateAttr()
    
    def __init__(self, few_shots_dir: str = "data/few_shots", **kwargs):
        super().__init__(few_shots_dir=few_shots_dir, **kwargs)

        if not os.path.exists(few_shots_dir):
           logger.warning(f"Few-shot 样本目录不存在: {few_shots_dir}")

        self._selector = SampleSelector(few_shots_dir)
    
    def _run(self, design_input: Dict[str, Any], k: int = 3) -> list:
        try:
            best_samples = self._selector.select_best_k(design_input, k=k)
            logger.info(f"成功选择 {len(best_samples)} 个相似样本")
            return best_samples
        except Exception as e:
            logger.error(f"样本选择失败: {e}")
            return []
    
    async def _arun(self, design_input: Dict[str, Any], k: int = 3) -> list:
        return self._run(design_input, k)