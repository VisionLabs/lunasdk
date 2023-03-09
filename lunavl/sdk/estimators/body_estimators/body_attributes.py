"""Module contains an body attributes estimator

See headwear_.
"""
from enum import Enum
from typing import Iterable, List, Optional, Type, Union, Dict, TypeVar, Generic

from FaceEngine import HumanAttributeRequest, LowerBodyClothing  # pylint: disable=E0611,E0401

from lunavl.sdk.base import BaseEstimation

from ...async_task import AsyncTask, DefaultPostprocessingFactory
from ..base import BaseEstimator
from ..estimators_utils.extractor_utils import validateInputByBatchEstimator
from .bodywarper import BodyWarp, BodyWarpedImage


class ApparentGenderEnum(Enum):
    """Apparent gender enum"""

    Female = 0
    Male = 1
    Unknown = 2

    @classmethod
    def fromCoreGender(cls, coreGender) -> "ApparentGenderEnum":
        """
        Get enum element by core gender.

        Args:
            coreGender: core estimated gender

        Returns:
            corresponding gender
        """
        return cls[coreGender.name]


class ApparentGender(BaseEstimation):
    """
    Class for apparent gender. Apparent gender are markers such as physical build, voice, clothes, ...

    Estimation properties:

        - male
        - female
        - unknown
        - predominantGender
    """

    @property
    def male(self) -> float:
        """
        Get `male` predict value.

        Returns:
            value in range [0, 1]
        """
        return self._coreEstimation.male

    @property
    def female(self):
        """
        Get `female` predict value.

        Returns:
            value in range [0, 1]
        """
        return self._coreEstimation.female

    @property
    def unknown(self):
        """
        Get `unknown` predict value.

        Returns:
            value in range [0, 1]
        """
        return self._coreEstimation.unknown

    @property
    def predominantGender(self) -> ApparentGenderEnum:
        """Get apparent gender predominant state"""
        return ApparentGenderEnum.fromCoreGender(self._coreEstimation.result)

    def asDict(self) -> dict:
        """
        Convert to dict.

        Returns:
            dict in platform format
        """
        return {
            "predominant_gender": str(self._coreEstimation.result.name).lower(),
            "estimations": {
                "male": self.male,
                "female": self.female,
                "unknown": self.unknown,
            },
        }


class BackpackStateEnum(Enum):
    """Backpack state enum"""

    No = 0  # backpack not in place
    Yes = 1  # backpack in place
    Unknown = 2  # unknown state

    @classmethod
    def fromCoreBackpack(cls, coreBackpack) -> "BackpackStateEnum":
        """
        Get enum element by core backpack.

        Args:
            coreBackpack: core backpack state enum

        Returns:
            corresponding backpack
        """
        return cls[coreBackpack.name]


class BackpackState(BaseEstimation):
    """
    Class for backpack state estimation.

    Headwear state properties:

        - yes
        - no
        - unknown
        - predominantState
    """

    @property
    def yes(self) -> float:
        """
        Get `yes` predict value.

        Returns:
            value in range [0, 1]
        """
        return self._coreEstimation.backpack

    @property
    def no(self):
        """
        Get `no` predict value.

        Returns:
            value in range [0, 1]
        """
        return self._coreEstimation.noBackpack

    @property
    def unknown(self):
        """
        Get `unknown` predict value.

        Returns:
            value in range [0, 1]
        """
        return self._coreEstimation.unknown

    @property
    def predominantState(self) -> BackpackStateEnum:
        """Get backpack predominant state"""
        return BackpackStateEnum.fromCoreBackpack(self._coreEstimation.result)

    def asDict(self) -> dict:
        """
        Convert to dict.

        Returns:
            dict in platform format
        """
        return {
            "predominant_state": str(self._coreEstimation.result.name).lower(),
            "estimations": {
                "yes": self.yes,
                "no": self.no,
                "unknown": self.unknown,
            },
        }


class HeadwearStateEnum(Enum):
    """Headwear state enum"""

    No = 0  # headwear not in place
    Yes = 1  # headwear in place
    Unknown = 2  # unknown state

    @classmethod
    def fromCoreHeadwear(cls, coreHeadwear) -> "HeadwearStateEnum":
        """
        Get enum element by core headwear.

        Args:
            coreHeadwear: core headwear state enum

        Returns:
            corresponding headwear
        """
        return cls[coreHeadwear.name]


class SleeveLength(Enum):
    """Sleeve length enum"""

    Short = 0  # short sleeve length
    Long = 1  # long sleeve length
    Unknown = 2  # unknown sleeve length

    @classmethod
    def fromCoreSleeve(cls, coreSleeve) -> "SleeveLength":
        """
        Get enum element by core sleeve size.

        Args:
            coreSleeve: core sleeve state enum

        Returns:
            corresponding sleeve
        """
        return cls[coreSleeve.name]


class Sleeve(BaseEstimation):
    """
    Class for sleeve estimation.

    Sleeve properties:

        - long
        - short
        - unknown
        - predominantState
    """

    @property
    def long(self) -> float:
        """
        Get `long` predict value.

        Returns:
            value in range [0, 1]
        """
        return self._coreEstimation.longSize

    @property
    def short(self):
        """
        Get `short` predict value.

        Returns:
            value in range [0, 1]
        """
        return self._coreEstimation.shortSize

    @property
    def unknown(self):
        """
        Get `unknown` predict value.

        Returns:
            value in range [0, 1]
        """
        return self._coreEstimation.unknown

    @property
    def predominantState(self) -> SleeveLength:
        """Get predominant state"""
        return SleeveLength.fromCoreSleeve(self._coreEstimation.result)

    def asDict(self) -> dict:
        """
        Convert to dict.

        Returns:
            dict in platform format
        """
        return {
            "predominant_state": str(self._coreEstimation.result.name).lower(),
            "estimations": {
                "long": self.long,
                "short": self.short,
                "unknown": self.unknown,
            },
        }


class AttributesPallette(Enum):
    """Colors enum"""

    Beige = "beige"
    Black = "black"
    Blue = "blue"
    Brown = "brown"
    Green = "green"
    Grey = "grey"
    Khaki = "khaki"
    Multicolored = "multicolored"
    Orange = "orange"
    Pink = "pink"
    Purple = "purple"
    Red = "red"
    White = "white"
    Yellow = "yellow"
    Unknown = "unknown"


OutwearColorEnum = AttributesPallette
LowerGarmentColor = AttributesPallette


class OutwearColor(BaseEstimation):
    """
    Class for outwear color estimation.

    Estimation properties:
        - colors
    """

    @property
    def colors(self) -> List[OutwearColorEnum]:
        """Get all find colors"""
        res = []

        if self._coreEstimation.isBeige:
            res.append(OutwearColorEnum.Beige)
        if self._coreEstimation.isBlack:
            res.append(OutwearColorEnum.Black)
        if self._coreEstimation.isBlue:
            res.append(OutwearColorEnum.Blue)
        if self._coreEstimation.isBrown:
            res.append(OutwearColorEnum.Brown)
        if self._coreEstimation.isGreen:
            res.append(OutwearColorEnum.Green)
        if self._coreEstimation.isGrey:
            res.append(OutwearColorEnum.Grey)
        if self._coreEstimation.isKhaki:
            res.append(OutwearColorEnum.Khaki)
        if self._coreEstimation.isMulticolored:
            res.append(OutwearColorEnum.Multicolored)
        if self._coreEstimation.isOrange:
            res.append(OutwearColorEnum.Orange)
        if self._coreEstimation.isPurple:
            res.append(OutwearColorEnum.Purple)
        if self._coreEstimation.isPink:
            res.append(OutwearColorEnum.Pink)
        if self._coreEstimation.isRed:
            res.append(OutwearColorEnum.Red)
        if self._coreEstimation.isWhite:
            res.append(OutwearColorEnum.White)
        if self._coreEstimation.isYellow:
            res.append(OutwearColorEnum.Yellow)
        if not res:
            res.append(OutwearColorEnum.Unknown)
        return res

    def asDict(self) -> List[str]:
        """
        Convert to list.

        Returns:
            list of colors
        """
        return [color.value for color in self.colors]


def asDict(x):
    """Wraps object asDict(). Returns None in case of no object."""
    if x is None:
        return None

    return x.asDict()


def nullable(cls: Type, estimation) -> Optional[object]:
    """
    Ensure validation is valid before creating resulting object.
    """
    if not estimation.isValid():
        return None

    return cls(estimation.value())


class LowerGarmentType(Enum):
    """Garment type enum."""

    Unknown = "unknown"
    Skirt = "skirt"
    Shorts = "shorts"
    Trousers = "trousers"


ColorTypeVar = TypeVar("ColorTypeVar")


class ColorMixin(Generic[ColorTypeVar]):
    """Extracts color from core estimation."""

    _colors: Dict[str, ColorTypeVar] = {}

    def singleColor(self, estimation, unknown: ColorTypeVar) -> ColorTypeVar:
        """Find first method (ie isBlack) that returns true, and return appropriate color name."""
        for key, value in self._colors.items():
            if getattr(estimation, key):
                return value
        return unknown

    def multiColor(self, estimation, unknown: ColorTypeVar) -> list[ColorTypeVar]:
        """Find all methods (ie isBlack) that return true, and return appropriate color name."""
        result = []
        for key, value in self._colors.items():
            if getattr(estimation, key):
                result.append(value)

        if not result:
            result.append(unknown)
        return result


class HeadwearColor(Enum):
    """Headwear color enum."""

    Black = "black"
    White = "white"
    Other = "other"
    Unknown = "unknown"


class HeadwearState(BaseEstimation, ColorMixin[HeadwearColor]):
    """
    Class for Headwear state estimation.

    Headwear state properties:

        - yes
        - no
        - unknown
        - predominantState
    """

    _colors = {
        "isBlack": HeadwearColor.Black,
        "isWhite": HeadwearColor.White,
        "isOther": HeadwearColor.Other,
        "isUnknown": HeadwearColor.Unknown,
    }

    @property
    def yes(self) -> float:
        """
        Get `yes` predict value.

        Returns:
            value in range [0, 1]
        """
        return self._coreEstimation.hat

    @property
    def no(self):
        """
        Get `no` predict value.

        Returns:
            value in range [0, 1]
        """
        return self._coreEstimation.noHat

    @property
    def unknown(self):
        """
        Get `unknown` predict value.

        Returns:
            value in range [0, 1]
        """
        return self._coreEstimation.unknown

    @property
    def predominantState(self) -> HeadwearStateEnum:
        """Get headwear predominant state"""
        return HeadwearStateEnum.fromCoreHeadwear(self._coreEstimation.result)

    @property
    def apparentColor(self) -> HeadwearColor:
        return self.singleColor(self._coreEstimation.hatColor, unknown=HeadwearColor.Unknown)

    def asDict(self) -> dict:
        """
        Convert to dict.

        Returns:
            dict in platform format
        """
        return {
            "apparent_color": self.apparentColor.value,
            "predominant_state": str(self._coreEstimation.result.name).lower(),
            "estimations": {
                "yes": self.yes,
                "no": self.no,
                "unknown": self.unknown,
            },
        }


class ShoesColor(Enum):
    """Shoes color enum."""

    Black = "black"
    White = "white"
    Other = "other"
    Unknown = "unknown"


class Shoes(BaseEstimation, ColorMixin[ShoesColor]):
    """Shoes estimation."""

    _colors = {
        "isBlack": ShoesColor.Black,
        "isWhite": ShoesColor.White,
        "isOther": ShoesColor.Other,
        "isUnknown": ShoesColor.Unknown,
    }

    @property
    def apparentColor(self) -> ShoesColor:
        """Return name of the estimated shoe color."""
        return self.singleColor(self._coreEstimation, unknown=ShoesColor.Unknown)

    def asDict(self) -> dict:
        """Serialize to dict."""
        return {"apparent_color": self.apparentColor.value}


class LowerGarment(BaseEstimation, ColorMixin[LowerGarmentColor]):
    """Lower garment estimantion."""

    _colors = {
        "isBlack": LowerGarmentColor.Black,
        "isWhite": LowerGarmentColor.White,
        "isBlue": LowerGarmentColor.Blue,
        "isGreen": LowerGarmentColor.Green,
        "isGrey": LowerGarmentColor.Grey,
        "isOrange": LowerGarmentColor.Orange,
        "isPurple": LowerGarmentColor.Purple,
        "isRed": LowerGarmentColor.Red,
        "isYellow": LowerGarmentColor.Yellow,
        "isPink": LowerGarmentColor.Pink,
        "isBrown": LowerGarmentColor.Brown,
        "isBeige": LowerGarmentColor.Beige,
        "isKhaki": LowerGarmentColor.Khaki,
        "isMulticolored": LowerGarmentColor.Multicolored,
    }
    _types = {
        LowerBodyClothing.Unknown: LowerGarmentType.Unknown,
        LowerBodyClothing.Skirt: LowerGarmentType.Skirt,
        LowerBodyClothing.Pants: LowerGarmentType.Trousers,
        LowerBodyClothing.Shorts: LowerGarmentType.Shorts,
    }

    @property
    def type(self) -> LowerGarmentType:
        """Type of the garment."""
        return self._types[self._coreEstimation.result]

    @property
    def colors(self) -> list[LowerGarmentColor]:
        """List of colors with scores exceed threshold."""
        if self.type == "undefined":
            return [LowerGarmentColor.Unknown]
        return self.multiColor(self._coreEstimation.lowerBodyClothingColor, unknown=LowerGarmentColor.Unknown)

    def asDict(self) -> dict:
        """Serialize to dict."""
        return {"type": self.type.value, "colors": [color.value for color in self.colors]}


class BodyAttributes(BaseEstimation):
    """
    Container for estimated body attributes.

    Attributes:
        apparentAge (Optional[float]): apparent age markers such as physical build, voice, clothes, and hair
        apparentGender (Optional[ApparentGender]): apparent gender markers such as physical build, voice, clothes,
          and hair
        backpack (Optional[BackpackState]): backpack state (yes, unknown or not)
        headwear (Optional[HeadwearState]): backpack state (yes, unknown or not)
        outwearColor (Optional[OutwearColor]): outwear color list
        sleeve (Optional[Sleeve]): sleeve size estimation
    """

    __slots__ = (
        "apparentAge",
        "apparentGender",
        "backpack",
        "headwear",
        "outwearColor",
        "sleeve",
        "lowerGarment",
        "shoes",
    )

    def __init__(self, coreEstimation):
        super().__init__(coreEstimation)

        if not coreEstimation.age_opt.isValid():
            self.apparentAge = None
        else:
            self.apparentAge = coreEstimation.age_opt.value()

        if not coreEstimation.gender_opt.isValid():
            self.apparentGender = None
        else:
            self.apparentGender = ApparentGender(coreEstimation.gender_opt.value())

        if not coreEstimation.backpack_opt.isValid():
            self.backpack = None
        else:
            self.backpack = BackpackState(coreEstimation.backpack_opt.value())

        self.headwear = nullable(HeadwearState, coreEstimation.headwear_opt)

        if not coreEstimation.outwearColor_opt.isValid():
            self.outwearColor = None
        else:
            self.outwearColor = OutwearColor(coreEstimation.outwearColor_opt.value())

        if not coreEstimation.sleeve_opt.isValid():
            self.sleeve = None
        else:
            self.sleeve = Sleeve(coreEstimation.sleeve_opt.value())

        self.lowerGarment = nullable(LowerGarment, coreEstimation.lowerBodyClothing_opt)
        self.shoes = nullable(Shoes, coreEstimation.shoeColor_opt)

    def asDict(self) -> Union[dict, list]:
        """Convert to dict"""
        return {
            "apparent_age": round(self.apparentAge) if self.apparentAge is not None else None,
            "apparent_gender": self.apparentGender.asDict() if self.apparentGender is not None else None,
            "backpack": self.backpack.asDict() if self.backpack is not None else None,
            "headwear": asDict(self.headwear),
            "sleeve": self.sleeve.asDict() if self.sleeve is not None else None,
            "outwear_color": self.outwearColor.asDict() if self.outwearColor is not None else None,
            "lower_garment": asDict(self.lowerGarment),
            "shoes": asDict(self.shoes),
        }


POST_PROCESSING = DefaultPostprocessingFactory(BodyAttributes)


class BodyAttributesEstimator(BaseEstimator):
    """
    Body attributes estimator.
    """

    #  pylint: disable=W0221
    def estimate(  # type: ignore
            self, warp: Union[BodyWarp, BodyWarpedImage], asyncEstimate: bool = False
    ) -> Union[BodyAttributes, AsyncTask[BodyAttributes]]:
        """
        Estimate body attributes on warp.

        Args:
            warp: warped image
            asyncEstimate: estimate or run estimation in background

        Returns:
            estimated body attributes if asyncEstimate is false otherwise async task
        Raises:
            LunaSDKException: if estimation failed
        """
        if asyncEstimate:
            task = self._coreEstimator.asyncEstimate(warp.warpedImage.coreImage, HumanAttributeRequest.EstimateAll)
            return AsyncTask(task, POST_PROCESSING.postProcessing)
        error, headwear = self._coreEstimator.estimate(warp.warpedImage.coreImage, HumanAttributeRequest.EstimateAll)
        return POST_PROCESSING.postProcessing(error, headwear)

    #  pylint: disable=W0221
    def estimateBatch(
            self, warps: List[Union[BodyWarp, BodyWarpedImage]], asyncEstimate: bool = False
    ) -> Union[List[BodyAttributes], AsyncTask[List[BodyAttributes]]]:
        """
        Batch estimate body attributes

        Args:
            warps: warped images
            asyncEstimate: estimate or run estimation in background
        Returns:
            list of estimated body attributes if asyncEstimate is false otherwise async task
        Raises:
            LunaSDKException: if estimation failed
        """
        coreImages = [warp.warpedImage.coreImage for warp in warps]

        validateInputByBatchEstimator(self._coreEstimator, coreImages, HumanAttributeRequest.EstimateAll)
        if asyncEstimate:
            task = self._coreEstimator.asyncEstimate(coreImages, HumanAttributeRequest.EstimateAll)
            return AsyncTask(task, POST_PROCESSING.postProcessingBatch)
        error, estimations = self._coreEstimator.estimate(coreImages, HumanAttributeRequest.EstimateAll)
        return POST_PROCESSING.postProcessingBatch(error, estimations)

    def aggregate(self, attributes: Iterable[BodyAttributes]) -> BodyAttributes:
        """Aggregate several body attributes to one"""
        return POST_PROCESSING.postProcessing(
            *self._coreEstimator.aggregate(
                [attribute.coreEstimation for attribute in attributes], HumanAttributeRequest.EstimateAll
            )
        )
