import copy
import json
import os
import random
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
import requests
import numpy as np

PAYMENT_STORE_KEY = 'session1_payment'  # Key for participant.vars to store payment-related data
AMOUNT_PATTERN = re.compile(r'[-+]?\d+(?:\.\d+)?')  # Regex to extract numeric amounts

rng = random.SystemRandom()  # independent RNG to avoid external seeding effects

LOTTERIES_PATH = Path(__file__).with_name('lotteries.json')
PRACTICE_LOTTERY_PATH = Path(__file__).with_name('practice_lottery.json')
TURNSTILE_VERIFY_URL = 'https://challenges.cloudflare.com/turnstile/v0/siteverify'


def _constants():
    from . import Constants

    return Constants




def validate_turnstile(token, secret, remoteip=None):
    data = {
        'secret': secret,
        'response': token
    }

    if remoteip:
        data['remoteip'] = remoteip

    try:
        response = requests.post(TURNSTILE_VERIFY_URL, data=data, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Turnstile validation error: {e}")
        return {'success': False, 'error-codes': ['internal-error']}


def verify_turnstile_token(token, remoteip=None, secret=None):
    if not token:
        return False, {'success': False, 'error-codes': ['missing-input-response']}
    if secret is None:
        secret = os.getenv('TURNSTILE_SECRET') or ''
    if not secret:
        return False, {'success': False, 'error-codes': ['missing-input-secret']}
    payload = {
        'secret': secret,
        'response': token,
    }
    if remoteip:
        payload['remoteip'] = remoteip
    try:
        response = requests.post(TURNSTILE_VERIFY_URL, data=payload, timeout=10)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        print(f"Turnstile validation error: {exc}")
        return False, {'success': False, 'error-codes': ['internal-error']}
    return bool(data.get('success')), data

def _assign_lottery_node_ids(lottery_id, lottery):
    """Attach stable node ids for rendering, without changing labels."""
    periods = (lottery or {}).get('periods', {})
    if not isinstance(periods, dict):
        return
    for period_key, nodes in periods.items():
        if not isinstance(nodes, list):
            continue
        for idx, node in enumerate(nodes):
            if not isinstance(node, dict):
                continue
            if node.get('id'):
                continue
            node['id'] = f"{lottery_id}:{period_key}:{idx}"


def load_lotteries():
    with LOTTERIES_PATH.open(encoding='utf-8') as handle:
        data = json.load(handle)
    for lottery_id, lottery in data.items():
        periods = lottery.get('periods')
        if not isinstance(periods, dict):
            continue
        converted = {}
        for key, value in periods.items():
            try:
                key = int(key)
            except (TypeError, ValueError):
                pass
            converted[key] = value
        lottery['periods'] = converted
        _assign_lottery_node_ids(lottery_id, lottery)
    return data



def load_test_lotteries():
    with PRACTICE_LOTTERY_PATH.open(encoding='utf-8') as handle:
        data = json.load(handle)
    for lottery_id, lottery in data.items():
        periods = lottery.get('periods')
        if not isinstance(periods, dict):
            continue
        converted = {}
        for key, value in periods.items():
            try:
                key = int(key)
            except (TypeError, ValueError):
                pass
            converted[key] = value
        lottery['periods'] = converted
        _assign_lottery_node_ids(lottery_id, lottery)
    return data




def _lottery_outcome_count(lottery):
    """Return outcome count as int when possible."""
    try:
        return int((lottery or {}).get('outcome_number'))
    except (TypeError, ValueError):
        return None


def _is_low_stake(lottery):
    return (lottery or {}).get('stake') == 'lo'


def build_lottery_order(lotteries):
    """Randomize lottery order while forcing a low-stake 4-outcome first round."""
    lottery_ids = list((lotteries or {}).keys())
    if not lottery_ids:
        return []

    first_candidates = [
        lottery_id
        for lottery_id, meta in (lotteries or {}).items()
        if _is_low_stake(meta) and _lottery_outcome_count(meta) == 4
    ]
    if first_candidates:
        first_id = rng.choice(first_candidates)
    else:
        first_id = rng.choice(lottery_ids)

    remaining = [lottery_id for lottery_id in lottery_ids if lottery_id != first_id]
    rng.shuffle(remaining)
    return [first_id] + remaining


def get_participant_lottery_order(participant):
    """Return (and store) a participant-specific lottery order."""
    order = participant.vars.get('lottery_order')
    if not order:
        Constants = _constants()
        order = build_lottery_order(Constants.lotteries)
        participant.vars['lottery_order'] = order
    return order


def determine_lottery_blank(lottery):
    stake = lottery.get('stake') or 0
    return True if stake == 'hi' else False


def is_high_stake(lottery):
    return (lottery or {}).get('stake') == 'hi'


def cutoff_validation_errors(values, lottery):
    errors = {}
    if values.get('cutoff_index') in (None, ''):
        errors['cutoff_index'] = 'Please select a cutoff.'
    if is_high_stake(lottery):
        if values.get('fine_cutoff_index') in (None, ''):
            errors['fine_cutoff_index'] = 'Please select a refined cutoff.'
    return errors or None


def parse_payoff_label(label):
    """Convert textual payoff labels such as '+10' into integers."""
    if not label:
        return None
    cleaned = str(label).replace(',', '')
    # Normalize common unicode minus/dash characters to ASCII minus.
    for token in ('\u2212', '\u2013', '\u2014', '\u2012', '\u2011'):
        cleaned = cleaned.replace(token, '-')
    # Strip everything except digits, sign, and decimal point.
    cleaned = re.sub(r'[^0-9+\-\.]', '', cleaned)
    match = AMOUNT_PATTERN.search(cleaned)
    if not match:
        return None
    try:
        value = float(match.group())
    except ValueError:
        return None
    return int(round(value))


def format_payoff_value(value):
    """Format numeric payoff values using the lottery currency convention."""
    if value is None:
        return None
    sign = '+' if value >= 0 else '-'
    # Use a unicode escape to avoid mojibake in HTML templates.
    return f"{sign}\u00a3{abs(int(round(value)))}"


def _weighted_choice(nodes):
    """Sample a node according to the 'probability' field."""
    if not nodes:
        return None
    weights = []
    total = 0.0
    for node in nodes:
        prob = node.get('probability')
        try:
            prob = float(prob)
        except (TypeError, ValueError):
            prob = 0.0
        weights.append(prob)
        total += prob
    if total <= 0:  # If no probabilities defined, choose uniformly
        return nodes[rng.randrange(len(nodes))]
    draw = rng.random() * total
    cumulative = 0
    for node, weight in zip(nodes, weights):  # Iterate through nodes and find the drawn one
        cumulative += weight  # Update cumulative probability
        if draw <= cumulative:  # Check if drawn value falls within this node's range
            return node  # If none matched, return the last node
    return nodes[-1]


def _sample_period_node(period_nodes, parent_label):
    """Return a deep copy of a randomly selected node that descends from parent_label."""
    parent_label = parent_label or 'Start'
    candidates = [
        node for node in period_nodes
        if (node.get('from') or 'Start') == parent_label
    ]  # Filter nodes by parent
    chosen = _weighted_choice(candidates)
    return copy.deepcopy(chosen) if chosen else None


def get_player_field(player, field_name):
    """Safely retrieve a possibly-null model field."""
    getter = getattr(player, 'field_maybe_none', None)
    if callable(getter):
        return getter(field_name)
    return getattr(player, field_name, None)


def get_payment_store(player):
    """Shortcut for accessing the participant-level storage used across rounds."""
    store = player.participant.vars.get(PAYMENT_STORE_KEY)
    if store is None:
        store = {}
        player.participant.vars[PAYMENT_STORE_KEY] = store
    return store


def persist_payment_store(player, store):
    """Persist payment store updates immediately."""
    player.participant.vars[PAYMENT_STORE_KEY] = store


def ensure_payment_lottery_selected(player):
    """Select the paying round/lottery once (after the main evaluation rounds)."""
    Constants = _constants()
    store = get_payment_store(player)
    if store.get('selected_round'):
        if 'selected_lottery_name' not in player.participant.vars:
            lottery_name = store.get('lottery_name')
            if lottery_name:
                player.participant.vars['selected_lottery_name'] = lottery_name
        return store

    paying_round = rng.randint(1, Constants.initial_evaluation_rounds)
    paying_player = player.in_round(paying_round)
    lottery_id = paying_player.lottery_id or Constants.default_lottery
    base_lottery = Constants.lotteries.get(lottery_id, Constants.lotteries[Constants.default_lottery])

    store.update(
        selected_round=paying_round,
        lottery_id=lottery_id,
        lottery_name=base_lottery.get('name'),
        selected_option=get_player_field(paying_player, 'selected_option'),
        selected_amount=get_player_field(paying_player, 'selected_amount'),
        selected_choice=get_player_field(paying_player, 'selected_choice'),
        cutoff_index=get_player_field(paying_player, 'cutoff_index'),
    )
    player.participant.vars['selected_lottery_name'] = store.get('lottery_name')

    for round_no in Constants.continuation_rounds:
        future_player = player.in_round(round_no)
        future_player.lottery_id = lottery_id

    persist_payment_store(player, store)
    return store


def ensure_payment_setup(player):
    """Guarantee that the payment selection has been made, even from later rounds."""
    Constants = _constants()
    store = get_payment_store(player)
    if store.get('selected_round'):
        return store
    anchor_round = min(player.round_number, Constants.initial_evaluation_rounds)
    anchor_player = player.in_round(anchor_round)
    return ensure_payment_lottery_selected(anchor_player)


def ensure_realized_up_to(player, period, store=None):
    """Realize the lottery path sequentially up to the requested period (1-based)."""
    Constants = _constants()
    store = store or get_payment_store(player)
    lottery_id = store.get('lottery_id') or player.lottery_id or Constants.default_lottery
    base_lottery = Constants.lotteries.get(lottery_id, Constants.lotteries[Constants.default_lottery])
    periods = base_lottery.get('periods', {})
    realized = store.setdefault('realized_nodes', {})
    changed = False

    if period >= 1 and 1 not in realized:
        node = _sample_period_node(periods.get(1, []), 'Start')
        if node:
            node['realized'] = True
            realized[1] = node
            changed = True

    if period >= 2 and 1 in realized and 2 not in realized:
        node = _sample_period_node(periods.get(2, []), realized[1]['label'])
        if node:
            node['realized'] = True
            realized[2] = node
            changed = True

    if period >= 3 and 2 in realized and 3 not in realized:
        node = _sample_period_node(periods.get(3, []), realized[2]['label'])
        if node:
            node['realized'] = True
            realized[3] = node
            changed = True

    store['realized_nodes'] = realized
    if changed:
        persist_payment_store(player, store)
    return store


def get_selected_lottery(player, store=None):
    """Return a deep copy of the base lottery selected for payment."""
    Constants = _constants()
    store = store or ensure_payment_setup(player)
    lottery_id = store.get('lottery_id') or player.lottery_id or Constants.default_lottery
    base_lottery = Constants.lotteries.get(lottery_id, Constants.lotteries[Constants.default_lottery])
    return copy.deepcopy(base_lottery)


def build_conditional_lottery(base_lottery, realized_nodes):
    """Create a conditional lottery focused on the remaining path."""
    realized_nodes = realized_nodes or {}
    periods = base_lottery.get('periods', {})
    truncated_periods = {}
    allowed_parents = {'Start'}

    for idx in sorted(periods.keys()):
        if idx == 0:
            truncated_periods[idx] = [copy.deepcopy(node) for node in periods[idx]]
            allowed_parents = {node.get('label') for node in truncated_periods[idx]}
            continue

        if idx in realized_nodes:
            realized_node = copy.deepcopy(realized_nodes[idx])
            parent_label = next(iter(allowed_parents or {'Start'}))
            realized_node['from'] = parent_label
            realized_node['probability'] = 1
            realized_node['abs_prob'] = 1
            realized_node['realized'] = True
            truncated_periods[idx] = [realized_node]
            allowed_parents = {realized_node.get('label')}
            continue

        filtered = [
            copy.deepcopy(node)
            for node in periods[idx]
            if not allowed_parents or (node.get('from') or 'Start') in allowed_parents
        ]
        truncated_periods[idx] = filtered
        allowed_parents = {node.get('label') for node in filtered}

    conditional = copy.deepcopy(base_lottery)
    conditional['periods'] = truncated_periods

    if truncated_periods:
        last_period = max(truncated_periods.keys())
        final_nodes = truncated_periods.get(last_period, [])
        unrealized_final_nodes = [node for node in final_nodes if not node.get('realized')]
        target_nodes = unrealized_final_nodes or final_nodes
        numeric_values = []
        for node in target_nodes:
            value = parse_payoff_label(node.get('label'))
            if value is not None:
                numeric_values.append(value)
        if numeric_values:
            conditional['max_payoff'] = max(numeric_values)
            conditional['min_payoff'] = min(numeric_values)
        else:
            conditional['max_payoff'] = base_lottery.get('max_payoff')
            conditional['min_payoff'] = base_lottery.get('min_payoff')
        conditional['outcome_number'] = len(target_nodes)
    return conditional


def get_conditional_lottery(player, realized_up_to):
    """Shortcut for retrieving the conditional lottery for continuation rounds."""
    store = ensure_payment_setup(player)
    realized_nodes = {
        period: copy.deepcopy(node)
        for period, node in store.get('realized_nodes', {}).items()
        if period <= realized_up_to
    }
    base_lottery = get_selected_lottery(player, store=store)
    return build_conditional_lottery(base_lottery, realized_nodes)


def compute_final_payoff(store):
    """Determine the final payoff based on the participant's choice."""
    if not store:
        return None
    Constants = _constants()
    selected_option = store.get('selected_option')
    if selected_option == Constants.options[0]:
        return store.get('selected_amount')
    realized = store.get('realized_nodes', {})
    final_node = realized.get(3)
    if not final_node:
        return None
    return parse_payoff_label(final_node.get('label'))


def compute_realized_offset(realized_nodes, upto_period=None):
    """Sum realized payoffs up to the specified period."""
    if not realized_nodes:
        return 0
    total = 0
    for period, node in realized_nodes.items():
        if upto_period is not None and period > upto_period:
            continue
        value = parse_payoff_label(node.get('label'))
        if value is not None:
            total += value
    return total


def compute_upcoming_payoff_range(lottery, start_period):
    """Return min/max totals from start_period to the final period."""
    periods = (lottery or {}).get('periods', {}) or {}
    if not periods or start_period not in periods:
        return None, None
    last_period = max(periods.keys())

    def _collect_totals(period, parent_labels):
        nodes = periods.get(period, [])
        if parent_labels is not None:
            nodes = [
                node for node in nodes
                if (node.get('from') or 'Start') in parent_labels
            ]
        if not nodes:
            return []
        totals = []
        for node in nodes:
            value = parse_payoff_label(node.get('label'))
            if value is None:
                continue
            if period >= last_period:
                totals.append(value)
                continue
            child_totals = _collect_totals(period + 1, {node.get('label')})
            if child_totals:
                totals.extend([value + child for child in child_totals])
            else:
                totals.append(value)
        return totals

    totals = _collect_totals(start_period, None)
    if not totals:
        return None, None
    return min(totals), max(totals)


def with_upcoming_payoff_range(lottery, start_period):
    """Clone lottery with min/max set to upcoming payoff range."""
    min_payoff, max_payoff = compute_upcoming_payoff_range(lottery, start_period)
    if min_payoff is None or max_payoff is None:
        return lottery
    adjusted = dict(lottery)
    adjusted['min_payoff'] = min_payoff
    adjusted['max_payoff'] = max_payoff
    return adjusted


def choice_amounts(player, lottery, base_offset=0):
    Constants = _constants()
    stake = lottery.get('stake') or 0
    if stake == 'lo':
        if player.round_number not in Constants.continuation_rounds:
            count = Constants.short_choice_count_first
        else:
            count = Constants.short_choice_count_later
    else:
        count = Constants.long_choice_count
    amounts = np.linspace(
        base_offset + lottery['min_payoff'],
        base_offset + lottery['max_payoff'],
        count,
        endpoint=True,
    )
    rounded = [int(round(a)) for a in amounts]
    # Drop duplicate rows caused by rounding so the table shows unique values.
    unique = []
    last = object()
    for value in rounded:
        if value == last:
            continue
        unique.append(value)
        last = value
    return unique


def choice_field_names(player, lottery, base_offset=0):
    count = len(choice_amounts(player, lottery, base_offset))
    return [f'chf_{i}' for i in range(1, count + 1)]


def choice_rows(player, lottery, base_offset=0):
    Constants = _constants()
    amounts = choice_amounts(player, lottery, base_offset)
    field_names = choice_field_names(player, lottery, base_offset)
    stake = lottery.get('stake')
    refinement_count = Constants.high_refinement_count if stake == 'hi' else 0
    rows = []
    for idx, (amount, field_name) in enumerate(zip(amounts, field_names)):
        prev_amount = amounts[idx - 1] if idx > 0 else None
        refinement_series = []
        if refinement_count and prev_amount is not None:
            refinement_series = build_refinement_series(
                prev_amount,
                amount,
                refinement_count,
            )
        rows.append(
            {
                'index': idx,
                'amount': amount,
                'abs_amount': abs(amount),
                'field_name': field_name,
                'prev_amount': prev_amount,
                'refinement_series': refinement_series,
            }
        )
    return rows


def build_refined_amounts(lower, upper, count):
    """Return evenly spaced integer amounts between lower and upper (exclusive)."""
    if count is None or count <= 0:
        return []
    if lower is None or upper is None:
        return []
    if lower == upper:
        return []
    start = float(lower)
    end = float(upper)
    descending = False
    if start > end:
        start, end = end, start
        descending = True
    grid = np.linspace(start, end, count + 2, endpoint=True)[1:-1]
    values = [int(round(val)) for val in grid]
    return list(reversed(values)) if descending else values


def build_refinement_series(lower, upper, count):
    """Return ordered fine options strictly above lower and including upper."""
    if lower is None or upper is None or lower == upper:
        return []
    between = build_refined_amounts(lower, upper, count)
    candidates = list(between)
    candidates.append(int(round(upper)))
    ordered = []
    for value in candidates:
        if not ordered or ordered[-1] != value:
            ordered.append(value)
    return ordered


def should_show_bridge(player, expected_round, prefix):
    """Return True when the participant should remain on the bridge page."""
    if player.round_number != expected_round:
        return False
    start_ts, _ = get_session_start_info(player, prefix)
    if start_ts is None:
        return False
    return time.time() < start_ts


def build_bridge_context(player, current_session, next_session, prefix):
    """Prepare template data shared by both bridge pages."""
    start_ts, readable = get_session_start_info(player, prefix)
    return {
        'this_session': current_session,
        'next_session': next_session,
        'wait_timestamp': start_ts,
        'wait_readable': readable,
        'server_timestamp': time.time(),
    }


def build_play_context(player, choice_lottery, display_lottery=None, base_offset=0):
    """Shared template variables for the valuation pages."""
    Constants = _constants()
    display_lottery = display_lottery or choice_lottery
    choice_rows_data = choice_rows(player, choice_lottery, base_offset)
    display_periods = display_lottery.get('periods', {})
    return {
        'lottery': json.dumps(display_lottery),
        'lottery_name': display_lottery.get('name'),
        'outcome_number': display_lottery.get('outcome_number'),
        'lottery_description': display_lottery.get('description'),
        'max_payoff': choice_lottery.get('max_payoff'),
        'min_payoff': choice_lottery.get('min_payoff'),
        'choice_rows': choice_rows_data,
        'stake': choice_lottery.get('stake'),
        'high_refinement_count': Constants.high_refinement_count,
        'accept_label': Constants.options[0],
        'play_label': Constants.options[1],
        'period_1': display_periods.get(1, []),
        'period_2': display_periods.get(2, []),
        'period_3': display_periods.get(3, []),
        'display_lottery': json.dumps(display_lottery),
        'choice_lottery': json.dumps(choice_lottery),
        'base_offset': base_offset,
        'evaluation_rounds': Constants.initial_evaluation_rounds,
    }


def _wheel_node_identifier(period, node):
    """Build a stable identifier for a lottery node used in the fortune wheel."""
    if not isinstance(node, dict):
        return None
    node_id = node.get('id')
    if node_id:
        return str(node_id)
    label = node.get('label') or f'Outcome_{period}'
    parent = node.get('from') or 'Start'
    return f'{period}:{parent}:{label}'


def build_wheel_segments(lottery, period, realized_nodes=None):
    """Return the wheel segments for the specified lottery period."""
    realized_nodes = realized_nodes or {}
    periods = (lottery or {}).get('periods', {})
    nodes = periods.get(period, [])
    if not isinstance(nodes, list):
        return []

    if period == 1:
        parent_labels = {'Start'}
    else:
        prev = realized_nodes.get(period - 1) or {}
        parent_label = prev.get('label')
        parent_labels = {parent_label} if parent_label else None

    segments = []
    for idx, node in enumerate(nodes):
        if not isinstance(node, dict):
            continue
        parent = node.get('from') or 'Start'
        if parent_labels and parent not in parent_labels:
            continue
        label = node.get('label') or f'Outcome {idx + 1}'
        probability = node.get('probability')
        segment_id = _wheel_node_identifier(period, node) or f'{period}:{idx}'
        segments.append(
            {
                'id': segment_id,
                'label': label,
                'probability': probability,
                'value': label,
            }
        )
    return segments


def build_wheel_context(
    player,
    *,
    period,
    title,
    description,
    stage_label,
    status_text,
    button_label,
    next_step,
    auto_spin=False,
    auto_submit=False,
    auto_submit_delay=0,
):
    """Assemble template variables for the fortune wheel page."""
    Constants = _constants()
    store = ensure_payment_setup(player)
    ensure_realized_up_to(player, period, store=store)
    lottery = get_selected_lottery(player, store=store)
    realized_nodes = store.get('realized_nodes', {})
    segments = build_wheel_segments(lottery, period, realized_nodes)
    predetermined = realized_nodes.get(period) or {}
    missing_outcome = not predetermined or not segments
    if missing_outcome:
        status_text = 'Outcome data missing. Please inform the admin.'
    accumulated_value = 0
    has_accumulated = False
    for node in realized_nodes.values():
        value = parse_payoff_label((node or {}).get('label'))
        if value is None:
            continue
        accumulated_value += value
        has_accumulated = True
    accumulated = format_payoff_value(accumulated_value) if has_accumulated else None
    return {
        'wheel_title': title,
        'wheel_description': description,
        'wheel_stage_label': stage_label,
        'wheel_status_text': status_text,
        'wheel_spin_button': button_label,
        'wheel_next_step': next_step,
        'wheel_segments': segments,
        'wheel_result_label': predetermined.get('label'),
        'wheel_result_id': _wheel_node_identifier(period, predetermined),
        'wheel_result_value': predetermined.get('label'),
        'wheel_spin_disabled': missing_outcome,
        'auto_spin': auto_spin,
        'auto_submit': auto_submit,
        'auto_submit_delay': auto_submit_delay,
        'allow_respin': False,
        'wheel_result_field': None,
        'wheel_result_json_field': None,
        'wheel_period': period,
        'lottery_name': store.get('lottery_name') or lottery.get('name'),
        'continuation_rounds': Constants.continuation_rounds,
        'realized_period1': realized_nodes.get(1),
        'realized_period2': realized_nodes.get(2),
        'realized_accumulated': accumulated,
        'realized_period3': realized_nodes.get(3),
    }


def build_realized_display(store):
    """Produce helper structures for realized nodes and payoff text."""
    realized_nodes = store.get('realized_nodes', {}) or {}
    summary = []
    for period in sorted(realized_nodes.keys()):
        node = realized_nodes.get(period) or {}
        label = node.get('label') or ''
        probability = node.get('probability')
        probability_text = None
        if isinstance(probability, (int, float)):
            probability_text = f"{probability * 100:.0f}%"
        summary.append(
            {
                'period': period,
                'label': label,
                'probability_text': probability_text,
            }
        )
    final_payoff = store.get('final_payoff')
    final_payoff_text = str(final_payoff) if final_payoff is not None else None
    return summary, realized_nodes, final_payoff_text


def clear_fine_cutoff(player):
    """Reset all fine-grained cutoff metadata."""
    player.fine_cutoff_index = None
    player.fine_selected_choice = None
    player.fine_selected_amount = None
    player.fine_cutoff_amount = None


def populate_fine_cutoff(player, lottery, amounts, cutoff_idx):
    """Store the fine-grained cutoff choice for high-stake lotteries."""
    Constants = _constants()
    if lottery.get('stake') != 'hi' or cutoff_idx <= 0:
        clear_fine_cutoff(player)
        return

    lower = amounts[cutoff_idx - 1]
    upper = amounts[cutoff_idx]
    refined_series = build_refinement_series(lower, upper, Constants.high_refinement_count)
    if not refined_series:
        clear_fine_cutoff(player)
        return

    fine_idx_value = get_player_field(player, 'fine_cutoff_index')
    try:
        fine_idx = int(round(float(fine_idx_value)))
    except (TypeError, ValueError):
        fine_idx = None

    if fine_idx is None:
        clear_fine_cutoff(player)
        return

    fine_idx = max(0, min(len(refined_series) - 1, fine_idx))
    player.fine_cutoff_index = fine_idx
    player.fine_selected_choice = fine_idx + 1
    player.fine_selected_amount = refined_series[fine_idx]
    player.fine_cutoff_amount = refined_series[fine_idx - 1] if fine_idx > 0 else lower


def store_cutoff_choice(player, lottery, base_offset=0):
    """Record the participant's cutoff information for the provided lottery."""
    amounts = choice_amounts(player, lottery, base_offset)
    count = len(amounts)
    if count == 0:
        player.selected_choice = None
        player.selected_amount = None
        player.selected_option = None
        player.cutoff_amount = None
        return

    cutoff_index = get_player_field(player, 'cutoff_index')
    if cutoff_index is not None:
        try:
            cutoff_idx_int = int(round(float(cutoff_index)))
        except (TypeError, ValueError):
            cutoff_idx_int = None
    else:
        cutoff_idx_int = None

    if cutoff_idx_int is not None:
        idx = max(0, min(count - 1, cutoff_idx_int))
        player.cutoff_amount = amounts[idx - 1] if idx > 0 else amounts[idx]
        player.selected_choice = idx + 1
        player.selected_amount = amounts[idx]
        field_name = f'chf_{idx + 1}'
        player.selected_option = getattr(player, field_name, None)
        populate_fine_cutoff(player, lottery, amounts, idx)
    else:
        player.cutoff_amount = None
        player.selected_choice = None
        player.selected_amount = None
        player.selected_option = None
        clear_fine_cutoff(player)


def format_session_date(dt):
    """Return a locale-aware readable date without platform-specific strftime flags."""
    if not isinstance(dt, datetime):
        return None
    return f"{dt.strftime('%A')}, {dt.day} {dt.strftime('%B')}"


def schedule_session_start(player, prefix, wait_seconds, future_round):
    """Store the scheduled start time for the next session and propagate it."""
    Constants = _constants()
    participant = player.participant
    existing_ts = participant.vars.get(f'{prefix}_start')
    existing_readable = participant.vars.get(f'{prefix}_start_readable')
    if existing_ts is not None:
        if existing_readable is None:
            try:
                existing_readable = format_session_date(datetime.fromtimestamp(existing_ts))
            except (TypeError, OSError, ValueError):
                existing_readable = None
        setattr(player, f'{prefix}_start', existing_ts)
        if existing_readable is not None:
            setattr(player, f'{prefix}_start_readable', existing_readable)
            participant.vars[f'{prefix}_start_readable'] = existing_readable
        if future_round and future_round <= Constants.num_rounds:
            future_player = player.in_round(future_round)
            setattr(future_player, f'{prefix}_start', existing_ts)
            if existing_readable is not None:
                setattr(future_player, f'{prefix}_start_readable', existing_readable)
        return existing_ts, existing_readable

    t = datetime.now() + timedelta(seconds=wait_seconds)
    start_ts = t.timestamp()
    readable = format_session_date(t)
    setattr(player, f'{prefix}_start', start_ts)
    setattr(player, f'{prefix}_start_readable', readable)
    participant.vars[f'{prefix}_start'] = start_ts
    participant.vars[f'{prefix}_start_readable'] = readable
    if future_round and future_round <= Constants.num_rounds:
        future_player = player.in_round(future_round)
        setattr(future_player, f'{prefix}_start', start_ts)
        setattr(future_player, f'{prefix}_start_readable', readable)
    return start_ts, readable


def get_session_start_info(player, prefix):
    """Retrieve the stored start timestamp and readable string for a session prefix."""
    start_ts = get_player_field(player, f'{prefix}_start')
    readable = get_player_field(player, f'{prefix}_start_readable')
    if start_ts is None:
        start_ts = player.participant.vars.get(f'{prefix}_start')
    if readable is None:
        readable = player.participant.vars.get(f'{prefix}_start_readable')
    return start_ts, readable
