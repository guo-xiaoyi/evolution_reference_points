from otree.api import *
import copy
import json
import numpy as np
import random
import re
from datetime import datetime, timedelta
import time

doc = """
Compound lottery with three periods for player evaluation.
Players make decisions using a multiple choice list.
"""

PAYMENT_STORE_KEY = 'session1_payment' # Key for participant.vars to store payment-related data
AMOUNT_PATTERN = re.compile(r'[-+]?\d+(?:\.\d+)?') # Regex to extract numeric amounts

rng = random.SystemRandom()  # independent RNG to avoid external seeding effects

class Constants(BaseConstants):
    name_in_url = 'compound_lottery_session1'
    players_per_group = None
    num_rounds = 17
    initial_evaluation_rounds = 15
    continuation_rounds = (16, 17)
    
    
    options = ['Safe option', 'Risky option']
    short_choice_count = 10
    long_choice_count = 20
    max_choice_count = long_choice_count
    
    # Dictionary of lottery structures
    lotteries = {
        'lottery_1': {
            'name': 'Apple',
            'outcome_number': 4,
            'max_payoff': 40,
            'min_payoff': -12,
            'description': 'Example lottery with four final outcomes',
            'periods': {
                0: [{'label': 'Start', 'probability': 1, 'from': None, 'abs_prob' : 1}],
                1: [
                    {'label': '+$10', 'probability': 0.6, 'from': 'Start', 'abs_prob' : 0.6},
                    {'label': '-$10', 'probability': 0.4, 'from': 'Start', 'abs_prob' : 0.4}
                ],
                2: [
                    {'label': '+$7', 'probability': 1, 'from': '+$10', 'abs_prob' : 0.6},
                    {'label': '-$12', 'probability': 1, 'from': '-$10', 'abs_prob' : 0.4}
                ],
                3: [
                    {'label': '+$8', 'probability': 0.8, 'from': '+$7', 'parent': '+$10', 'abs_prob' : 0.48},
                    {'label': '+$0', 'probability': 0.2, 'from': '+$7', 'parent': '+$10', 'abs_prob' : 0.12},
                    {'label': '+$2', 'probability': 0.5, 'from': '-$12', 'parent': '-$10', 'abs_prob' : 0.2},
                    {'label': '+$5', 'probability': 0.5, 'from': '-$12', 'parent': '-$10', 'abs_prob' : 0.2},
                ]
            }
        },
        'lottery_2': {
            'name': 'Banana',
            'outcome_number': 6,
            'max_payoff': 825,
            'min_payoff': -1245,
            'description': 'Example lottery with four final outcomes',
            'periods': {
                0: [{'label': 'Start', 'probability': 1, 'from': None, 'abs_prob' : 1}],
                1: [
                    {'label': '+$610', 'probability': 0.7, 'from': 'Start', 'abs_prob' : 0.6},
                    {'label': '+$645', 'probability': 0.3, 'from': 'Start', 'abs_prob' : 0.4}
                ],
                2: [
                    {'label': '-$665', 'probability': 1, 'from': '+$610', 'abs_prob' : 0.6},
                    {'label': '-$895', 'probability': 0.6, 'from': '+$645', 'abs_prob' : 0.4},
                    {'label': '-$800', 'probability': 0.4, 'from': '+$645', 'abs_prob' : 0.4}
                ],
                3: [
                    {'label': '+$865', 'probability': 0.3, 'from': '-$665', 'parent': '+$610', 'abs_prob' : 0.48},
                    {'label': '-$925', 'probability': 0.7, 'from': '-$665', 'parent': '+$610', 'abs_prob' : 0.12},
                    {'label': '+$940', 'probability': 0.6, 'from': '-$895', 'parent': '+$645', 'abs_prob' : 0.2},
                    {'label': '-$995', 'probability': 0.4, 'from': '-$895', 'parent': '+$645', 'abs_prob' : 0.2},
                    {'label': '-$860', 'probability': 0.6, 'from': '-$800', 'parent': '+$645', 'abs_prob' : 0.2},
                    {'label': '+$980', 'probability': 0.4, 'from': '-$800', 'parent': '+$645', 'abs_prob' : 0.2}
                ]
            }
        },
    }

    # Default lottery to use
    default_lottery = 'lottery_1'


class Subsession(BaseSubsession):
    pass


class Group(BaseGroup):
    pass


class Player(BasePlayer):
    num_failed_attempts = models.IntegerField(initial=0)
    failed_too_many = models.BooleanField(initial=False)
    quiz1 = models.IntegerField(blank=True)
    quiz2 = models.IntegerField(blank=True)
    quiz3 = models.IntegerField(blank=True)
    quiz4 = models.IntegerField(blank=True)
    quiz5 = models.IntegerField(blank=True)

    # Lottery assigned per round in creating_session
    lottery_id = models.StringField()

    # Store the index of the cutoff selected in the UI
    cutoff_index = models.IntegerField(blank=True)
    cutoff_amount = models.IntegerField(blank=True)

    # Store the choice selected for payment
    selected_choice = models.IntegerField(blank=True)
    selected_amount = models.IntegerField(blank=True)
    
    # Store which option was chosen for the selected choice
    selected_option = models.StringField(blank=True)
    
    # Store the schedule start for session 2 and 3
    session2_start = models.FloatField(blank=True)
    session2_start_readable = models.StringField(blank=True)
    session3_start = models.FloatField(blank=True)
    session3_start_readable = models.StringField(blank=True)


# Dynamically add the multiple choice list fields (supports up to long list length)
for i in range(1, Constants.max_choice_count + 1): 
    setattr(
        Player,
        f'chf_{i}',
        models.StringField(choices=Constants.options, blank=True)
    )


def parse_payoff_label(label):
    """Convert textual payoff labels such as '+$10' into integers."""
    if not label:
        return None
    cleaned = label.replace(',', '').replace('$', '').strip()
    match = AMOUNT_PATTERN.search(cleaned)
    if not match:
        return None
    try:
        value = float(match.group())
    except ValueError:
        return None
    return int(round(value))


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
    if total <= 0: # If no probabilities defined, choose uniformly
        return nodes[rng.randrange(len(nodes))]
    draw = rng.random() * total
    cumulative = 0
    for node, weight in zip(nodes, weights): # Iterate through nodes and find the drawn one
        cumulative += weight # Update cumulative probability
        if draw <= cumulative: # Check if drawn value falls within this node's range
            return node# If none matched, return the last node
    return nodes[-1]


def _sample_period_node(period_nodes, parent_label):
    """Return a deep copy of a randomly selected node that descends from parent_label."""
    parent_label = parent_label or 'Start' 
    candidates = [
        node for node in period_nodes
        if (node.get('from') or 'Start') == parent_label
    ] # Filter nodes by parent
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


def ensure_payment_lottery_selected(player):
    """Select the paying round/lottery once (after the main evaluation rounds)."""
    store = get_payment_store(player)
    if store.get('selected_round'):
        return store

    paying_round = random.randint(1, Constants.initial_evaluation_rounds)
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

    for round_no in Constants.continuation_rounds:
        future_player = player.in_round(round_no)
        future_player.lottery_id = lottery_id

    ensure_realized_up_to(player, 1, store=store)
    return store


def ensure_payment_setup(player):
    """Guarantee that the payment selection has been made, even from later rounds."""
    store = get_payment_store(player)
    if store.get('selected_round'):
        return store
    anchor_round = min(player.round_number, Constants.initial_evaluation_rounds)
    anchor_player = player.in_round(anchor_round)
    return ensure_payment_lottery_selected(anchor_player)


def ensure_realized_up_to(player, period, store=None):
    """Realize the lottery path sequentially up to the requested period (1-based)."""
    store = store or get_payment_store(player)
    lottery_id = store.get('lottery_id') or player.lottery_id or Constants.default_lottery
    base_lottery = Constants.lotteries.get(lottery_id, Constants.lotteries[Constants.default_lottery])
    periods = base_lottery.get('periods', {})
    realized = store.setdefault('realized_nodes', {})

    if period >= 1 and 1 not in realized:
        node = _sample_period_node(periods.get(1, []), 'Start')
        if node:
            node['realized'] = True
            realized[1] = node

    if period >= 2 and 1 in realized and 2 not in realized:
        node = _sample_period_node(periods.get(2, []), realized[1]['label'])
        if node:
            node['realized'] = True
            realized[2] = node

    if period >= 3 and 2 in realized and 3 not in realized:
        node = _sample_period_node(periods.get(3, []), realized[2]['label'])
        if node:
            node['realized'] = True
            realized[3] = node

    store['realized_nodes'] = realized
    return store


def get_selected_lottery(player, store=None):
    """Return a deep copy of the base lottery selected for payment."""
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


# PAGES
class Welcome(Page):
    @staticmethod
    def is_displayed(player):
        return player.round_number == 1
    pass

class session1(Page):
    @staticmethod
    def is_displayed(player):
        return player.round_number == 1
    pass

class Introduction(Page):
    @staticmethod
    def is_displayed(player):
        return player.round_number == 1
    pass


class Play(Page):
    form_model = 'player'



    @staticmethod
    def is_displayed(player):
        return player.round_number in range(1, 16)

    @staticmethod
    def _choice_amounts(lottery, base_offset=0):
        outcome_number = lottery.get('outcome_number') or 0
        count = Constants.short_choice_count if outcome_number <= 4 else Constants.long_choice_count
        amounts = np.linspace(
            base_offset + lottery['min_payoff'],
            base_offset + lottery['max_payoff'],
            count,
            endpoint=True
        )
        return [int(round(a)) for a in amounts]

    @staticmethod
    def _choice_field_names(lottery, base_offset=0):
        count = len(Play._choice_amounts(lottery, base_offset))
        return [f'chf_{i}' for i in range(1, count + 1)]

    @staticmethod
    def _choice_rows(lottery, base_offset=0):
        amounts = Play._choice_amounts(lottery, base_offset)
        field_names = Play._choice_field_names(lottery, base_offset)
        return [
            {
                'index': idx,
                'amount': amount,
                'abs_amount': abs(amount),
                'field_name': field_name
            }
            for idx, (amount, field_name) in enumerate(zip(amounts, field_names))
        ]

    @staticmethod
    def get_form_fields(player):
        lottery = Constants.lotteries[player.lottery_id]
        fields = Play._choice_field_names(lottery, base_offset=0)
        fields.append('cutoff_index')
        return fields

    @staticmethod
    def vars_for_template(player):
        lottery = Constants.lotteries[player.lottery_id]
        return build_play_context(player, choice_lottery=lottery, display_lottery=lottery, base_offset=0)

    @staticmethod
    def before_next_page(player, timeout_happened):
        lottery = Constants.lotteries[player.lottery_id]
        store_cutoff_choice(player, lottery)
        if player.round_number == Constants.initial_evaluation_rounds:
            schedule_session_start(
                player,
                prefix='session2',
                wait_seconds=86400,
                future_round=Constants.continuation_rounds[0],
            )
            ensure_payment_lottery_selected(player)


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
    display_lottery = display_lottery or choice_lottery
    choice_rows = Play._choice_rows(choice_lottery, base_offset)
    display_periods = display_lottery.get('periods', {})
    return {
        'lottery': json.dumps(display_lottery),
        'lottery_name': display_lottery.get('name'),
        'outcome_number': display_lottery.get('outcome_number'),
        'lottery_description': display_lottery.get('description'),
        'max_payoff': choice_lottery.get('max_payoff'),
        'min_payoff': choice_lottery.get('min_payoff'),
        'choice_rows': choice_rows,
        'accept_label': Constants.options[0],
        'play_label': Constants.options[1],
        'period_1': display_periods.get(1, []),
        'period_2': display_periods.get(2, []),
        'period_3': display_periods.get(3, []),
        'display_lottery': json.dumps(display_lottery),
        'choice_lottery': json.dumps(choice_lottery),
        'base_offset': base_offset,
    }


def _wheel_node_identifier(period, node):
    """Build a stable identifier for a lottery node used in the fortune wheel."""
    if not isinstance(node, dict):
        return None
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
    store = ensure_payment_setup(player)
    ensure_realized_up_to(player, period, store=store)
    lottery = get_selected_lottery(player, store=store)
    realized_nodes = store.get('realized_nodes', {})
    segments = build_wheel_segments(lottery, period, realized_nodes)
    predetermined = realized_nodes.get(period) or {}

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


def store_cutoff_choice(player, lottery, base_offset=0):
    """Record the participant's cutoff information for the provided lottery."""
    amounts = Play._choice_amounts(lottery, base_offset)
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
    else:
        player.cutoff_amount = None
        player.selected_choice = None
        player.selected_amount = None
        player.selected_option = None


def schedule_session_start(player, prefix, wait_seconds, future_round):
    """Store the scheduled start time for the next session and propagate it."""
    t = datetime.now() + timedelta(seconds=wait_seconds)
    start_ts = t.timestamp()
    readable = t.strftime('%A, %B %d')
    setattr(player, f'{prefix}_start', start_ts)
    setattr(player, f'{prefix}_start_readable', readable)
    participant = player.participant
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


class Results(Page):
    @staticmethod
    def vars_for_template(player):
        return {
            'payment_level': player.selected_amount,
            'selected_option': player.selected_option
        }

class Check(Page):
    form_model = 'player'
    form_fields = ['quiz1', 'quiz2', 'quiz3', 'quiz4', 'quiz5']
    # This is for comprehension check
    @staticmethod
    def error_message(player: Player, values):
        solutions = dict(quiz1= 1, quiz2= 2, quiz3= 3, quiz4= 4, quiz5= 5)
        errors = {name : "Incorrect answer. Please try again." for name in solutions.keys()}
        if errors:
            player.num_failed_attempts += 1
            if player.num_failed_attempts >= 3:
                player.failed_too_many = True
            else:
                return errors
    
    @staticmethod
    def is_displayed(player):
        return player.round_number == 1


class Failed(Page):
    # This is for failed comprehension check
    @staticmethod
    def is_displayed(player: Player):
        return player.failed_too_many



class Draw(Page):
    # This page is only displayed in the final round to draw the lottery for payment.
    @staticmethod
    def is_displayed(player):
        return player.round_number == Constants.initial_evaluation_rounds

    @staticmethod
    def vars_for_template(player):
        store = ensure_payment_lottery_selected(player)
        full_lottery = get_selected_lottery(player, store=store)
        realized_summary, realized_nodes, final_payoff_text = build_realized_display(store)
        return {
            'selected_round': store.get('selected_round'),
            'selected_choice': store.get('selected_choice'),
            'selected_option': store.get('selected_option'),
            'selected_amount': store.get('selected_amount'),
            'lottery_name': store.get('lottery_name'),
            'lottery_id': store.get('lottery_id'),
            'selected_lottery': json.dumps(full_lottery),
            'continuation_rounds': Constants.continuation_rounds,
            'is_revision': False,
            'realized_summary': realized_summary,
            'realized_nodes_json': json.dumps(realized_nodes),
            'final_payoff_text': final_payoff_text,
            'current_session_number': None,
        }

    @staticmethod
    def before_next_page(player, timeout_happened):
        ensure_payment_lottery_selected(player)


class BridgeSession2(Page):
    @staticmethod
    def is_displayed(player):
        return should_show_bridge(player, Constants.initial_evaluation_rounds, 'session2')

    @staticmethod
    def vars_for_template(player):
        return build_bridge_context(player, current_session=1, next_session=2, prefix='session2')


class BridgeSession3(Page):
    @staticmethod
    def is_displayed(player):
        return should_show_bridge(player, Constants.continuation_rounds[0], 'session3')

    @staticmethod
    def vars_for_template(player):
        return build_bridge_context(player, current_session=2, next_session=3, prefix='session3')


class Play2(Page):
    form_model = 'player'

    @staticmethod
    def is_displayed(player):
        return player.round_number == Constants.continuation_rounds[0]

    @staticmethod
    def get_form_fields(player):
        store = ensure_payment_setup(player)
        ensure_realized_up_to(player, 1, store=store)
        lottery = get_conditional_lottery(player, realized_up_to=1)
        realized_nodes = store.get('realized_nodes', {})
        base_offset = compute_realized_offset(realized_nodes, upto_period=1)
        fields = Play._choice_field_names(lottery, base_offset=base_offset)
        fields.append('cutoff_index')
        return fields

    @staticmethod
    def vars_for_template(player):
        store = ensure_payment_setup(player)
        ensure_realized_up_to(player, 1, store=store)
        conditional_lottery = get_conditional_lottery(player, realized_up_to=1)
        full_lottery = get_selected_lottery(player, store=store)
        realized_nodes = store.get('realized_nodes', {})
        base_offset = compute_realized_offset(realized_nodes, upto_period=1)
        context = build_play_context(player, choice_lottery=conditional_lottery, display_lottery=full_lottery, base_offset=base_offset)
        realized_nodes = store.get('realized_nodes', {})
        context.update(
            continuation_stage=2,
            realized_period1=realized_nodes.get(1),
            realized_period2=None,
            selected_round=store.get('selected_round'),
            base_lottery_name=store.get('lottery_name'),
        )
        base_name = store.get('lottery_name') or context['lottery_name']
        context['lottery_name'] = f"{base_name} – continuation after period 1"
        return context

    @staticmethod
    def before_next_page(player, timeout_happened):
        store = ensure_payment_setup(player)
        lottery = get_conditional_lottery(player, realized_up_to=1)
        realized_nodes = store.get('realized_nodes', {})
        base_offset = compute_realized_offset(realized_nodes, upto_period=1)
        store_cutoff_choice(player, lottery, base_offset=base_offset)
        ensure_realized_up_to(player, 2, store=store)
        if player.round_number == Constants.continuation_rounds[0]:
            schedule_session_start(
                player,
                prefix='session3',
                wait_seconds=86400,
                future_round=Constants.continuation_rounds[1],
            )

class Play3(Page):
    form_model = 'player'
    template_name = 'session1/Play3.html'

    @staticmethod
    def is_displayed(player):
        return player.round_number == Constants.continuation_rounds[1]

    @staticmethod
    def get_form_fields(player):
        store = ensure_payment_setup(player)
        ensure_realized_up_to(player, 2, store=store)
        lottery = get_conditional_lottery(player, realized_up_to=2)
        realized_nodes = store.get('realized_nodes', {})
        base_offset = compute_realized_offset(realized_nodes, upto_period=2)
        fields = Play._choice_field_names(lottery, base_offset=base_offset)
        fields.append('cutoff_index')
        return fields

    @staticmethod
    def vars_for_template(player):
        store = ensure_payment_setup(player)
        ensure_realized_up_to(player, 2, store=store)
        conditional_lottery = get_conditional_lottery(player, realized_up_to=2)
        full_lottery = get_selected_lottery(player, store=store)
        realized_nodes = store.get('realized_nodes', {})
        base_offset = compute_realized_offset(realized_nodes, upto_period=2)
        context = build_play_context(player, choice_lottery=conditional_lottery, display_lottery=full_lottery, base_offset=base_offset)
        realized_nodes = store.get('realized_nodes', {})
        context.update(
            continuation_stage=3,
            realized_period1=realized_nodes.get(1),
            realized_period2=realized_nodes.get(2),
            realized_period3=realized_nodes.get(3),
            selected_round=store.get('selected_round'),
            base_lottery_name=store.get('lottery_name'),
        )
        base_name = store.get('lottery_name') or context['lottery_name']
        context['lottery_name'] = f"{base_name} – continuation after period 2"
        return context

    @staticmethod
    def before_next_page(player, timeout_happened):
        store = ensure_payment_setup(player)
        lottery = get_conditional_lottery(player, realized_up_to=2)
        realized_nodes = store.get('realized_nodes', {})
        base_offset = compute_realized_offset(realized_nodes, upto_period=2)
        store_cutoff_choice(player, lottery, base_offset=base_offset)
        ensure_realized_up_to(player, 3, store=store)
        final_payoff = compute_final_payoff(store)
        if final_payoff is not None:
            store['final_payoff'] = final_payoff
            realized_nodes = store.get('realized_nodes', {})
            final_node = realized_nodes.get(3)
            if final_node:
                store['final_outcome_label'] = final_node.get('label')
            player.participant.vars['session1_final_payoff'] = final_payoff


class Revision_session2(Page):
    # This page is only displayed in the continuation rounds to show the drawn lottery.
    template_name = 'session1/Draw.html'
    @staticmethod
    def is_displayed(player):
        return player.round_number == Constants.continuation_rounds[0]

    @staticmethod
    def vars_for_template(player):
        store = ensure_payment_lottery_selected(player)
        full_lottery = get_selected_lottery(player, store=store)
        realized_summary, realized_nodes, final_payoff_text = build_realized_display(store)

        return {
            'selected_round': store.get('selected_round'),
            'selected_choice': store.get('selected_choice'),
            'selected_option': store.get('selected_option'),
            'selected_amount': store.get('selected_amount'),
            'lottery_name': store.get('lottery_name'),
            'lottery_id': store.get('lottery_id'),
            'realized_summary': realized_summary,
            'realized_nodes_json': json.dumps(realized_nodes),
            'final_payoff_text': final_payoff_text,
            'selected_lottery': json.dumps(full_lottery),
            'continuation_rounds': Constants.continuation_rounds,
            'is_revision': True,
            'current_session_number': 2 if player.round_number == Constants.continuation_rounds[0] else 3,
        }

class Revision_session3(Page):
    # This page is only displayed in the continuation rounds to show the drawn lottery.
    template_name = 'session1/Draw.html'
    @staticmethod
    def is_displayed(player):
        return player.round_number == Constants.continuation_rounds[1]

    @staticmethod
    def vars_for_template(player):
        store = ensure_payment_lottery_selected(player)
        full_lottery = get_selected_lottery(player, store=store)
        realized_summary, realized_nodes, final_payoff_text = build_realized_display(store)

        return {
            'selected_round': store.get('selected_round'),
            'selected_choice': store.get('selected_choice'),
            'selected_option': store.get('selected_option'),
            'selected_amount': store.get('selected_amount'),
            'lottery_name': store.get('lottery_name'),
            'lottery_id': store.get('lottery_id'),
            'realized_summary': realized_summary,
            'realized_nodes_json': json.dumps(realized_nodes),
            'final_payoff_text': final_payoff_text,
            'selected_lottery': json.dumps(full_lottery),
            'continuation_rounds': Constants.continuation_rounds,
            'is_revision': True,
            'current_session_number': 2 if player.round_number == Constants.continuation_rounds[0] else 3,
        }

class WheelSession2(Page):
    """Fortune wheel to realize the first continuation outcome (for Session 2)."""
    template_name = 'session1/fortune_wheel.html'

    @staticmethod
    def is_displayed(player):
        return player.round_number == Constants.initial_evaluation_rounds

    @staticmethod
    def vars_for_template(player):
        return build_wheel_context(
            player,
            period=1,
            title='Spin for Session 2',
            description='We now realize the first period of your lottery. The outcome applies to Session 2 (Play2).',
            stage_label='Period 1 outcome',
            status_text='Click the button to reveal the branch you will face in Session 2 (Play2).',
            button_label='Spin for Session 2',
            next_step='This outcome determines the branch you will evaluate in Session 2 (Play2).',
        )


class WheelSession3(Page):
    """Fortune wheel to realize the second continuation outcome (for Session 3)."""
    template_name = 'session1/fortune_wheel.html'

    @staticmethod
    def is_displayed(player):
        return player.round_number == Constants.continuation_rounds[0]

    @staticmethod
    def vars_for_template(player):
        return build_wheel_context(
            player,
            period=2,
            title='Spin for Session 3',
            description='We now realize the second period of your lottery. The outcome applies to Session 3 (Play3).',
            stage_label='Period 2 outcome',
            status_text='Click the button to reveal the branch you will face in Session 3 (Play3).',
            button_label='Spin for Session 3',
            next_step='This outcome determines the branch you will evaluate in Session 3 (Play3).',
        )


class Thankyou(Page):
    @staticmethod
    def is_displayed(player):
        return player.round_number == Constants.continuation_rounds[1]


class WelcomeSession2(Page):
    @staticmethod
    def is_displayed(player):
        return player.round_number == Constants.continuation_rounds[0]
    
class WelcomeSession3(Page):
    @staticmethod
    def is_displayed(player):
        return player.round_number == Constants.continuation_rounds[1]

class session2(Page):
    @staticmethod
    def is_displayed(player):
        return player.round_number == Constants.continuation_rounds[0]

class session3(Page):
    @staticmethod
    def is_displayed(player):
        return player.round_number == Constants.continuation_rounds[1]

class Revision_session1(Page):
    template_name = 'session1/Draw.html'
    @staticmethod
    def is_displayed(player):
        return player.round_number == Constants.initial_evaluation_rounds
    
    @staticmethod
    def vars_for_template(player):
        store = ensure_payment_lottery_selected(player)
        full_lottery = get_selected_lottery(player, store=store)
        realized_summary, realized_nodes, final_payoff_text = build_realized_display(store)

        return {
            'selected_round': store.get('selected_round'),
            'selected_choice': store.get('selected_choice'),
            'selected_option': store.get('selected_option'),
            'selected_amount': store.get('selected_amount'),
            'lottery_name': store.get('lottery_name'),
            'lottery_id': store.get('lottery_id'),
            'realized_summary': realized_summary,
            'realized_nodes_json': json.dumps(realized_nodes),
            'final_payoff_text': final_payoff_text,
            'selected_lottery': json.dumps(full_lottery),
            'continuation_rounds': Constants.continuation_rounds,
            'is_revision': True,
            'current_session_number': 2 if player.round_number == Constants.continuation_rounds[0] else 3,
        }



page_sequence = [
    Welcome,
    session1,
    Introduction,
    Check,
    Failed,
    Play,
    Draw,
    WheelSession2,
    Revision_session1,
    BridgeSession2,
    WelcomeSession2,
    session2,
    Revision_session2,
    Play2,
    WheelSession3,
    BridgeSession3,
    WelcomeSession3,
    Revision_session3,
    session3,
    Play3,
    Thankyou,
]


def creating_session(subsession: Subsession):
    """Assign lotteries by round so each round uses a different definition."""
    lottery_ids = list(Constants.lotteries.keys())
    if not lottery_ids:
        return
    idx = (subsession.round_number - 1) % len(lottery_ids)
    lottery_for_round = lottery_ids[idx]
    for player in subsession.get_players():
        player.lottery_id = lottery_for_round
