from __future__ import unicode_literals

ACTION_GRAPH_FILE='actionGraphs.pkl'
OP_SIG_MODEL_FILE='opSigModel.pkl'
RAW_MATERIAL_MODEL_FILE='rawMaterialModel.pkl'
PART_COMP_MODEL_FILE='partCompositeModel.pkl'
APP_MODEL_FILE='apparatusModel.pkl'


MATERIAL_TAG = 'material'
APPARATUS_TAG = 'apparatus'
INTERMEDIATE_PRODUCT_TAG = 'intrmed'
IMPLICIT_ARGUMENT_TAG = 'IMPLICIT_ARG'

LEAF_INDEX = -1

# Path to data JSON files.
data_path = u'/iesl/canvas/smysore/material_science_ag/papers_data_json'

# Token labels as used in the annotations; of use for the evaluations of the final graph outputs.
token_classes = ['null', 'amt_unit', 'amt_misc', 'cnd_unit', 'cnd_misc',
                 'material', 'target', 'operation', 'descriptor',
                 'prop_unit', 'prop_type', 'synth_aprt', 'char_aprt',
                 'brand', 'intrmed', 'number', 'meta', 'ref', 'prop_misc',
                 'cnd_type', 'aprt_unit', 'aprt_des']

older_token_classes = [u'null', u'amt_unit', u'amt_misc', u'cnd_unit', u'cnd_misc',
                       u'material', u'target', u'operation', u'descriptor',
                       u'prop_unit', u'prop_type', u'synth_aprt', u'char_aprt',
                       u'brand', u'intrmed', u'number', u'meta', u'ref', u'prop_misc']

token_class_dict = dict(zip(token_classes, range(len(token_classes))))
