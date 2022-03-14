from typing import TypeVar, Generic, Callable, Generator, List, Type, Optional

from FaceEngine import FSDKErrorResult  # pylint: disable=E0611,E0401

from lunavl.sdk.errors.exceptions import assertError

TaskResult = TypeVar("TaskResult")
CoreTaskResult = TypeVar("CoreTaskResult")


def postProcessingBatch(
    error: FSDKErrorResult, coreEstimations: List[CoreTaskResult], resultClass: Type[TaskResult]
) -> List[TaskResult]:
    """
    Post processing batch estimation
    Args:
        error:  estimation error
        coreEstimations:  core batch estimation
        resultClass: result class

    Returns:
        list of lunavl structure based on core estimation
    """
    assertError(error)

    return [resultClass(coreEstimation) for coreEstimation in coreEstimations]


def postProcessingBatchWithAggregation(
    error: FSDKErrorResult,
    coreEstimations: List[CoreTaskResult],
    aggregatedAttribute: CoreTaskResult,
    resultClass: Type[TaskResult],
    aggregate: bool,
) -> tuple[List[TaskResult], Optional[TaskResult]]:
    """
    Post processing batch estimation
    Args:
        error:  estimation error
        coreEstimations:  core batch estimation
        aggregatedAttribute: aggregated core estimation
        aggregate: need or not aggregate result
        resultClass: result class

    Returns:
        list of lunavl structure based on core estimation + aggregated estimation
    """
    assertError(error)

    estimations = [resultClass(coreEstimation) for coreEstimation in coreEstimations]
    if aggregate:
        return estimations, resultClass(aggregatedAttribute)
    else:
        return estimations, None


def postProcessing(error: FSDKErrorResult, coreEstimation: CoreTaskResult, resultClass: Type[TaskResult]) -> TaskResult:
    assertError(error)
    return resultClass(coreEstimation)


class DefaultPostprocessingFactory(Generic[TaskResult]):
    """
    Default estimation post processing factory.

    Factory is intended for generation standard post procesing  core estimation object (conversion to lunavl structure
    and check errors)

    Attributes:
        resultClass: result class
    """

    def __init__(self, resultClass: Type[TaskResult]):
        self.resultClass = resultClass

    def postProcessingBatch(self, error: FSDKErrorResult, coreEstimations: List[CoreTaskResult]) -> List[TaskResult]:
        """
        Post processing batch estimation
        Args:
            error: estimation error
            coreEstimations: core batch estimation

        Returns:
            list of lunavl structure based on core estimation
        """
        return postProcessingBatch(error, coreEstimations, resultClass=self.resultClass)

    def postProcessingBatchWithAggregation(
        self,
        error: FSDKErrorResult,
        coreEstimations: List[CoreTaskResult],
        aggregatedAttribute: CoreTaskResult,
        aggregate: bool,
    ) -> tuple[List[TaskResult], Optional[TaskResult]]:
        """
        Post processing batch estimation
        Args:
            error: estimation error
            coreEstimations: core batch estimation
            aggregate: need or not aggregate result

            aggregatedAttribute: aggregated core estimation

        Returns:
            list of lunavl structure based on core estimation + aggregated estimation
        """
        return postProcessingBatchWithAggregation(
            error=error,
            coreEstimations=coreEstimations,
            aggregatedAttribute=aggregatedAttribute,
            aggregate=aggregate,
            resultClass=self.resultClass,
        )

    def postProcessing(self, error: FSDKErrorResult, coreEstimation: CoreTaskResult) -> TaskResult:
        """
        Postprocessing single core estimation
        Args:
            error: estimation error
            coreEstimation: core batch estimation

        Returns:
            list of lunavl structure based on core estimation
        """
        return postProcessing(error=error, coreEstimation=coreEstimation, resultClass=self.resultClass)


class AsyncTask(Generic[TaskResult]):
    """
    SDK async task wrapper

    Attributes:
        coreTask (CoreAsyncTask): core task
        postProcessing (Callable): post processing callback for getting a task result in correct format
    """

    __slots__ = ["coreTask", "postProcessing"]

    def __init__(
        self, coreTask: "CoreAsyncTask", postProcessing: Callable[..., TaskResult]  # type: ignore # noqa: F821
    ):
        self.coreTask = coreTask
        self.postProcessing = postProcessing

    def get(self) -> TaskResult:
        """
        Join to thread and wait result
        Returns:
            task result
        """
        res = self.coreTask.get()
        return self.postProcessing(*res)

    def __await__(self) -> Generator[None, None, TaskResult]:
        """
        Await task
        Returns:
            task result
        """
        yield from self.coreTask.__await__()
        res = self.coreTask.getResult()
        return self.postProcessing(*res)
