from __future__ import annotations

from typing import Any

from .models import DiagnosisInput, DiagnosisReport


class CrewAINotInstalled(RuntimeError):
    pass


def build_crewai_flow(request: DiagnosisInput) -> Any:
    """延迟构建生产用CrewAI Flow，使离线测试不依赖CrewAI。

    返回的Flow使用Crew进行自主调查，并用Flow固定分诊和验证边界。
    场景工具及Pydantic结构化输出仍由本包负责。
    """
    try:
        from crewai import Agent, Crew, Process, Task
        from crewai.flow.flow import Flow, listen, start
    except ImportError as exc:
        raise CrewAINotInstalled(
            "请先安装RepoMedic集成依赖：pip install -e '.[repomedic]'"
        ) from exc

    class CrewAIRepoMedicFlow(Flow[dict[str, Any]]):
        @start()
        def triage(self) -> dict[str, Any]:
            self.state["request"] = request.model_dump(mode="json")
            return self.state

        @listen(triage)
        def investigate(self, state: dict[str, Any]) -> str:
            investigator = Agent(
                role="代码仓库调查员",
                goal="在给定仓库中寻找有证据支持的候选根因",
                backstory="必须引用文件和行号，绝不能声称看过实际未检查的证据。",
                verbose=False,
            )
            verifier = Agent(
                role="独立验证员",
                goal="质疑候选根因，只接受测试或源码证据支持的结论",
                backstory="独立于调查员，优先使用可以证伪的检查。",
                verbose=False,
            )
            investigation = Task(
                description=(
                    f"诊断仓库 {request.repository} 中的问题：{request.issue}\n"
                    f"Traceback：\n{request.traceback}"
                ),
                expected_output="包含文件、行号、证据和验证方案的JSON候选根因",
                agent=investigator,
            )
            verification = Task(
                description="审查调查结果，拒绝没有证据支持的候选根因。",
                expected_output="包含证据引用的JSON验证结论",
                agent=verifier,
                context=[investigation],
            )
            crew = Crew(
                agents=[investigator, verifier],
                tasks=[investigation, verification],
                process=Process.sequential,
                verbose=False,
            )
            return str(crew.kickoff())

    return CrewAIRepoMedicFlow()


def crewai_result_to_report(_result: Any) -> DiagnosisReport:
    candidate = getattr(_result, "pydantic", None)
    if isinstance(candidate, DiagnosisReport):
        return candidate
    candidate = getattr(_result, "json_dict", None)
    if candidate is not None:
        return DiagnosisReport.model_validate(candidate)
    if isinstance(_result, dict):
        return DiagnosisReport.model_validate(_result)
    raw = getattr(_result, "raw", _result)
    if isinstance(raw, str):
        return DiagnosisReport.model_validate_json(raw)
    raise ValueError("CrewAI结果中不包含结构化DiagnosisReport")
