#  Copyright (c) 2022. Terrier Team at University of Glasgow, http://http://terrierteam.dcs.gla.ac.uk
#
#  This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
#  If a copy of the MPL was not distributed with this  file, you can obtain one at
#  http://mozilla.org/MPL/2.0/.

from finance_pilot.data.filter.customer.customer_filter import CustomerFilter
from finance_pilot.utils.constants import DEFAULT_USER_COL
from pandas import DataFrame


class CustomerInTrain(CustomerFilter):
    """
    Only keeps those customers in the training se.
    """

    def filter(
        self,
        customers: set,
        time_series: DataFrame,
        train: DataFrame,
        valid: DataFrame,
        test: DataFrame,
    ) -> set:
        return set(train[DEFAULT_USER_COL].unique().flatten())
