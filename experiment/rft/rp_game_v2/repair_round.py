exec(r'''
SESSION_CODE = 'i2g9x805'
TARGET_CODES = []
DRY_RUN = False

from otree.database import db
from otree.models import Participant, Session
from otree.lookup import _get_session_lookups
from rp_game_v2 import Constants

session = db.query(Session).filter_by(code=SESSION_CODE).one()
lookups = _get_session_lookups(SESSION_CODE)
actual_max = max(lookups)

def page_idx(page_name, round_number):
    hits = [
        i for i, info in lookups.items()
        if info.app_name == 'rp_game_v2'
        and info.round_number == round_number
        and info.page_class.__name__ == page_name
    ]
    if len(hits) != 1:
        raise RuntimeError((page_name, round_number, hits))
    return hits[0]

bridge2_idx = page_idx('BridgeSession2', Constants.initial_evaluation_rounds)
bridge3_idx = page_idx('BridgeSession3', Constants.continuation_rounds[0])

s2_keys = {'selected_amount_s2', 'cutoff_index_s2', 'cutoff_amount_s2',
           'fine_selected_amount_s2', 'fine_cutoff_amount_s2'}
s3_keys = {'selected_amount_s3', 'cutoff_index_s3', 'cutoff_amount_s3',
           'fine_selected_amount_s3', 'fine_cutoff_amount_s3', 'final_payoff'}

participants = (
    db.query(Participant)
    .filter_by(session=session)
    .order_by(Participant.id_in_session)
    .all()
)

print('actual_max_page_index =', actual_max)
print('BridgeSession2 index =', bridge2_idx)
print('BridgeSession3 index =', bridge3_idx)

for p in participants:
    if p._max_page_index != actual_max:
        print('sync max:', p.code, p._max_page_index, '->', actual_max)
        if not DRY_RUN:
            p._max_page_index = actual_max

    if TARGET_CODES and p.code not in TARGET_CODES:
        continue

    is_out = p._index_in_pages > actual_max
    if not TARGET_CODES and not is_out:
        continue

    store = p.vars.get('session1_payment') or {}
    completed_s2 = any(store.get(k) is not None for k in s2_keys)
    completed_s3 = any(store.get(k) is not None for k in s3_keys)

    print('candidate:', p.code, 'label=', p.label, 'index=', p._index_in_pages,
          'max=', p._max_page_index, 'completed_s2=', completed_s2, 'completed_s3=', completed_s3)

    if completed_s3:
        print('skip completed:', p.code)
        continue

    target = bridge3_idx if (p.vars.get('session3_start') is not None or completed_s2) else bridge2_idx
    print('repair:', p.code, p._index_in_pages, '->', target)

    if not DRY_RUN:
        p._index_in_pages = target
        p.is_on_wait_page = False
        p._timeout_page_index = None
        p._timeout_expiration_time = None
        p.status = ''

if not DRY_RUN:
    db.commit()
    print('committed')
else:
    db.rollback()
    print('dry run only')
''')
