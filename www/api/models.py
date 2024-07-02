"""

"""

from django.db import models


class Relevance(models.Model):
    """

    >>> Relevance(
    ...     reference="I am interested in creating an SDOH Risk Index",
    ...     properties=[
    ...          Property(identity="Total Population"),
    ...          Property(identity="Total Population in Poverty"),
    ...     ],
    ... ).save()

    """

    id = models.BigAutoField(primary_key=True)
    reference: str = models.TextField()
    properties: list = models.JSONField()
    