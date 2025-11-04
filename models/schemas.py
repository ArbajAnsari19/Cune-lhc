from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class Applicant(BaseModel):
    name: Optional[str] = Field(default=None, description="Full name of the applicant")
    dateOfBirth: Optional[str] = Field(default=None, description="Date of birth in any provided format")
    fathersName: Optional[str] = Field(default=None, description="Father's name")
    address: Optional[str] = Field(default=None, description="Full address string")


class PolicyDetails(BaseModel):
    policyType: Optional[str] = None
    coverageAmount: Optional[str] = None
    planName: Optional[str] = None
    policyTerms: Optional[str] = None
    renewalBasis: Optional[str] = None


class MedicalDisclosure(BaseModel):
    medicalHistory: Optional[str] = None
    familyHistory: Optional[str] = None
    weight: Optional[str] = None
    height: Optional[str] = None
    currentMedication: Optional[str] = None


class LifestyleAssessment(BaseModel):
    smokingStatus: Optional[str] = None
    alcoholConsumption: Optional[str] = None
    physicalActivity: Optional[str] = None
    otherHabits: Optional[str] = None


class LifeSummary(BaseModel):
    applicant: Applicant
    policyDetails: PolicyDetails
    medicalDisclosure: MedicalDisclosure
    lifestyleAssessment: LifestyleAssessment


class PropertyDetails(BaseModel):
    propertyAddress: Optional[str] = None
    propertyType: Optional[str] = None
    constructionType: Optional[str] = None
    roofType: Optional[str] = None
    numberOfStories: Optional[int] = None
    yearBuilt: Optional[int] = None
    extras: Optional[Dict[str, Any]] = None


class PropertyFeatures(BaseModel):
    hasCentralAir: Optional[bool] = None
    hasFireExtinguisher: Optional[bool] = None
    hasSwimmingPool: Optional[bool] = None
    extras: Optional[Dict[str, Any]] = None


class RiskFactors(BaseModel):
    distanceToFireStation: Optional[str] = None
    floodZone: Optional[str] = None
    hasPreviousClaim: Optional[bool] = None
    extras: Optional[Dict[str, Any]] = None


class PropertyCasualtySummary(BaseModel):
    applicant: Applicant
    propertyDetails: PropertyDetails
    propertyFeatures: PropertyFeatures
    riskFactors: RiskFactors

