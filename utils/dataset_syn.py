# Copyright 2018 Timo Nolle
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# ==============================================================================

import gzip
import pickle as pickle

import numpy as np
import torch
from torch_geometric.data import Data
from utils.anomaly import label_to_targets
from utils.enums import AttributeType
from utils.enums import Class
from utils.enums import PadMode
from utils.fs import EventLogFile
from processmining.event import Event
from processmining.log import EventLog



class Dataset_syn(object):
    '''
        只为了同步属性值列表
    '''
    def __init__(self, dataset_name):
        # Public properties
        self.dataset_name = dataset_name


        # Load dataset
        if self.dataset_name is not None:
            self.load(self.dataset_name)

    def load(self, dataset_name):
        """
        Load dataset from disk.

        :param dataset_name:
        :return:
        """
        el_file = EventLogFile(dataset_name)
        self.dataset_name = el_file.name
        print(f'reading: {el_file.path}')
        # Else generate from event log
        if el_file.path.exists():
            self._event_log = EventLog.load(el_file.path)

            self.from_event_log(self._event_log)

        else:
            raise FileNotFoundError()

    def __len__(self):
        return len(self.case_lens)

    def from_event_log(self,event_log):
        include_attributes = event_log.event_attribute_keys

        feature_columns = dict(name=[])
        self.case_lens = []
        attr_types = event_log.get_attribute_types(include_attributes)

        # Create beginning of sequence event
        start_event = dict((a, EventLog.start_symbol if t == AttributeType.CATEGORICAL else 0.0) for a, t in
                           zip(include_attributes, attr_types))
        start_event = Event(timestamp=None, **start_event)

        # Create end of sequence event
        end_event = dict((a, EventLog.end_symbol if t == AttributeType.CATEGORICAL else 0.0) for a, t in
                         zip(include_attributes, attr_types))
        end_event = Event(timestamp=None, **end_event)

        # Save all values in a flat 1d array. This is necessary for the preprocessing. We will reshape later.
        for i, case in enumerate(event_log.cases):
            self.case_lens.append(case.num_events + 2)  # +2 for start and end events
            for event in [start_event] + case.events + [end_event]:
                for attribute in event_log.event_attribute_keys:
                    # Get attribute value from event log
                    if attribute == 'name':
                        attr = event.name
                    elif attribute in include_attributes:
                        attr = event.attributes[attribute]
                    else:
                        # Ignore the attribute name because its not part of included_attributes
                        continue

                    # Add to feature columns
                    if attribute not in feature_columns.keys():
                        feature_columns[attribute] = []
                    feature_columns[attribute].append(attr)

        self.unique_attrV={}
        for k,v in feature_columns.items():
            self.unique_attrV[k] = set(v)

        self.max_seq_len= max(self.case_lens)


