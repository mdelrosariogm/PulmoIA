"""
Esquemas Pydantic para validación de entrada/salida de la API.
"""

from enum import IntEnum

from pydantic import BaseModel, Field


class COPDLevel(IntEnum):
    BAJO_RIESGO = 0
    LEVE = 1
    MODERADO = 2
    GRAVE = 3
    MUY_GRAVE = 4


class AnalysisRequest(BaseModel):
    role: str = Field(..., description="Rol del usuario: 'medico' o 'paciente'")
    patient_name: str | None = Field(None, description="Nombre del paciente")
    patient_age: int | None = Field(None, ge=1, le=120)
    symptoms: list[str] | None = Field(default=[], description="Lista de síntomas reportados")


class AnalysisResult(BaseModel):
    copd_level: COPDLevel
    level_name: str
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confianza del modelo [0-1]")
    fev1_fvc: str = Field(..., description="Relación FEV1/FVC estimada")
    fev1_pct: str = Field(..., description="FEV1% estimado")
    pattern: str = Field(..., description="Patrón acústico detectado")
    recommendation: str
    emergency: bool = False
    features_used: int = Field(..., description="Número de features MFCC utilizados")
    model_version: str = "1.0.0"


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
    code: int
