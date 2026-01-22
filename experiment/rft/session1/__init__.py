from matplotlib.dviread import Page
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
    high_refinement_count = 5
    
    # Dictionary of lottery structures
    lotteries = {
        'lottery_1': {
            'name': 'Apple',
            'outcome_number': 4,
            'stake': 'lo',
            'max_payoff': 40,
            'min_payoff': -12,
            'description': 'Example lottery with four final outcomes',
            'periods': {
                0: [{'label': 'Start', 'probability': 1, 'from': None, 'abs_prob' : 1}],
                1: [
                    {'label': '+£10', 'probability': 0.6, 'from': 'Start', 'abs_prob' : 0.6},
                    {'label': '-£10', 'probability': 0.4, 'from': 'Start', 'abs_prob' : 0.4}
                ],
                2: [
                    {'label': '+£7', 'probability': 1, 'from': '+£10', 'abs_prob' : 0.6},
                    {'label': '-£12', 'probability': 1, 'from': '-£10', 'abs_prob' : 0.4}
                ],
                3: [
                    {'label': '+£8', 'probability': 0.8, 'from': '+£7', 'parent': '+£10', 'abs_prob' : 0.48},
                    {'label': '+£0', 'probability': 0.2, 'from': '+£7', 'parent': '+£10', 'abs_prob' : 0.12},
                    {'label': '+£2', 'probability': 0.5, 'from': '-£12', 'parent': '-£10', 'abs_prob' : 0.2},
                    {'label': '+£5', 'probability': 0.5, 'from': '-£12', 'parent': '-£10', 'abs_prob' : 0.2},
                ]
            }
        },

        'lottery_3': {
            'name': 'Lychee',
            'outcome_number': 4,
            'stake': 'hi',
            'max_payoff': 335,
            'min_payoff': -725,
            'description': 'Example lottery with four final outcomes',
            'periods': {
                0: [{'label': 'Start', 'probability': 1, 'from': None, 'abs_prob' : 1}],
                1: [
                    {'label': '+£120', 'probability': 0.8, 'from': 'Start', 'abs_prob' : 0.8},
                    {'label': '-£250', 'probability': 0.2, 'from': 'Start', 'abs_prob' : 0.2}
                ],
                2: [
                    {'label': '-£115', 'probability': 1, 'from': '+£120', 'abs_prob' : 0.8},
                    {'label': '£120', 'probability': 1, 'from': '-£250', 'abs_prob' : 0.2}
                ],
                3: [
                    {'label': '+£210', 'probability': 0.8, 'from': '-£115', 'parent': '+£120', 'abs_prob' : 0.64},
                    {'label': '-£625', 'probability': 0.2, 'from': '-£115', 'parent': '+£120', 'abs_prob' : 0.16},
                    {'label': '-£595', 'probability': 0.5, 'from': '£120', 'parent': '-£250', 'abs_prob' : 0.1},
                    {'label': '+£465', 'probability': 0.5, 'from': '£120', 'parent': '-£250', 'abs_prob' : 0.1},
                ]
            }
        },


        'lottery_2': {
            'name': 'Banana',
            'outcome_number': 6,
            'stake': 'hi',
            'max_payoff': 825,
            'min_payoff': -1245,
            'description': 'Example lottery with four final outcomes',
            'periods': {
                0: [{'label': 'Start', 'probability': 1, 'from': None, 'abs_prob' : 1}],
                1: [
                    {'label': '+£610', 'probability': 0.7, 'from': 'Start', 'abs_prob' : 0.6},
                    {'label': '+£645', 'probability': 0.3, 'from': 'Start', 'abs_prob' : 0.4}
                ],
                2: [
                    {'label': '-£665', 'probability': 1, 'from': '+£610', 'abs_prob' : 0.6},
                    {'label': '-£895', 'probability': 0.6, 'from': '+£645', 'abs_prob' : 0.4},
                    {'label': '-£800', 'probability': 0.4, 'from': '+£645', 'abs_prob' : 0.4}
                ],
                3: [
                    {'label': '+£865', 'probability': 0.3, 'from': '-£665', 'parent': '+£610', 'abs_prob' : 0.48},
                    {'label': '-£925', 'probability': 0.7, 'from': '-£665', 'parent': '+£610', 'abs_prob' : 0.12},
                    {'label': '+£940', 'probability': 0.6, 'from': '-£895', 'parent': '+£645', 'abs_prob' : 0.2},
                    {'label': '-£995', 'probability': 0.4, 'from': '-£895', 'parent': '+£645', 'abs_prob' : 0.2},
                    {'label': '-£860', 'probability': 0.6, 'from': '-£800', 'parent': '+£645', 'abs_prob' : 0.2},
                    {'label': '+£980', 'probability': 0.4, 'from': '-£800', 'parent': '+£645', 'abs_prob' : 0.2}
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
    failed_too_many_1 = models.BooleanField(initial=False)
    failed_too_many_2 = models.BooleanField(initial=False)
    failed_too_many_3 = models.BooleanField(initial=False)
    quiz1 = models.StringField(
        label='Do you need to be able to participate in all three sessions to participate in this experiment?',
        choices=['Yes', 'No'],
    )
    quiz2 = models.StringField(
        label='Is it fine to use AI in the experiment?',
        choices=['Yes', 'No'],
    )
    quiz3 = models.StringField(
        label='Imagine you make the following choice. Does this mean that you would rather receive the outcomes of the given random payoff tree than a payment of £4? ',
        choices=['Yes', 'No'],
    )
    quiz4 = models.StringField(
        label='Is the following sentence correct? The figure below signifies that there is a 60% chance of a gain of £10 and a 40% chance of a loss of £10.',
        choices=['Yes', 'No'],
    )
    quiz5 = models.StringField(
        label='Please consider the following random payoff tree. Imagine that the random determination of the first outcome (three days from now) yields a loss of £10, as indi-cated by the red arrow to “-£10”. Is it possible that the outcome six days from now yields a gain of £7? ',
        choices=['Yes', 'No'],
    )

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

    # Additional refinement information for high-stake lists
    fine_cutoff_index = models.IntegerField(blank=True)
    fine_selected_choice = models.IntegerField(blank=True)
    fine_selected_amount = models.IntegerField(blank=True)
    fine_cutoff_amount = models.IntegerField(blank=True)
    
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
    """Convert textual payoff labels such as '+£10' into integers."""
    if not label:
        return None
    cleaned = label.replace(',', '')
    for token in ('£', '¶œ', 'Ł', '稖'):
        cleaned = cleaned.replace(token, '')
    cleaned = cleaned.strip()
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
    return f"{sign}£{abs(int(round(value)))}"


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


def persist_payment_store(player, store):
    """Persist payment store updates immediately."""
    player.participant.vars[PAYMENT_STORE_KEY] = store


def ensure_payment_lottery_selected(player):
    """Select the paying round/lottery once (after the main evaluation rounds)."""
    store = get_payment_store(player)
    if store.get('selected_round'):
        if 'selected_lottery_name' not in player.participant.vars:
            lottery_name = store.get('lottery_name')
            if lottery_name:
                player.participant.vars['selected_lottery_name'] = lottery_name
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
    player.participant.vars['selected_lottery_name'] = store.get('lottery_name')

    for round_no in Constants.continuation_rounds:
        future_player = player.in_round(round_no)
        future_player.lottery_id = lottery_id

    persist_payment_store(player, store)
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


# PAGES
class Welcome(Page):
    @staticmethod
    def is_displayed(player):
        return player.round_number == 1
    pass

class Session1(Page):
    @staticmethod
    def is_displayed(player):
        return player.round_number == 1
    pass

class Introduction1(Page):
    @staticmethod
    def is_displayed(player):
        return player.round_number == 1
    pass

class Introduction2(Page):
    @staticmethod
    def is_displayed(player):
        return player.round_number == 1
    pass

class Introduction3(Page):
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
        stake = lottery.get('stake') or 0
        count = Constants.short_choice_count if stake == 'lo' else Constants.long_choice_count
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
                    refinement_count
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

    @staticmethod
    def get_form_fields(player):
        lottery = Constants.lotteries[player.lottery_id]
        choice_lottery = with_upcoming_payoff_range(lottery, start_period=1)
        fields = Play._choice_field_names(choice_lottery, base_offset=0)
        fields.append('cutoff_index')
        fields.append('fine_cutoff_index')
        return fields

    @staticmethod
    def vars_for_template(player):
        lottery = Constants.lotteries[player.lottery_id]
        choice_lottery = with_upcoming_payoff_range(lottery, start_period=1)
        return build_play_context(player, choice_lottery=choice_lottery, display_lottery=lottery, base_offset=0)

    @staticmethod
    def before_next_page(player, timeout_happened):
        lottery = Constants.lotteries[player.lottery_id]
        choice_lottery = with_upcoming_payoff_range(lottery, start_period=1)
        store_cutoff_choice(player, choice_lottery)
        if player.round_number == Constants.initial_evaluation_rounds:
            schedule_session_start(
                player,
                prefix='session2',
                wait_seconds=15,
                future_round=Constants.continuation_rounds[0],
            )
            ensure_payment_lottery_selected(player)


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
        populate_fine_cutoff(player, lottery, amounts, idx)
    else:
        player.cutoff_amount = None
        player.selected_choice = None
        player.selected_amount = None
        player.selected_option = None
        clear_fine_cutoff(player)


def schedule_session_start(player, prefix, wait_seconds, future_round):
    """Store the scheduled start time for the next session and propagate it."""
    participant = player.participant
    existing_ts = participant.vars.get(f'{prefix}_start')
    existing_readable = participant.vars.get(f'{prefix}_start_readable')
    if existing_ts is not None:
        if existing_readable is None:
            try:
                existing_readable = datetime.fromtimestamp(existing_ts).strftime('%A, %B %d')
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
    readable = t.strftime('%A, %B %d')
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


class Results(Page):
    @staticmethod
    def vars_for_template(player):
        return {
            'payment_level': player.selected_amount,
            'selected_option': player.selected_option
        }

class Check1(Page):
    allow_back_button = True
    form_model = 'player'
    form_fields = ['quiz1', 'quiz2']
    # This is for comprehension check
    @staticmethod
    def error_message(player: Player, values):
        solutions = dict(quiz1= 'Yes', quiz2= 'No')
        error_messages = {
            'quiz1': 'You should have chosen “Yes” for this question. Please only participate in this experiment if you can take part in all three sessions. Sessions two and three take place three and six days from now and will be much shorter than today’s session (only about 10 minutes each).',
            'quiz2': 'You should have chosen “No” for this question.  We are interested in your personal preferences (there are no right or wrong answers). Therefore, please do not use AI during the experiment.',
        }
        errors = {
            name: error_messages.get(name, 'Incorrect answer. Please try again.')
            for name in solutions
            if values[name] != solutions[name]
        }
        if errors:
            player.num_failed_attempts += 1
            if player.num_failed_attempts >= 5:
                player.failed_too_many_1 = True
            else:
                return errors
    
    @staticmethod
    def is_displayed(player):
        return player.round_number == 1

class Check2(Page):
    allow_back_button = True
    form_model = 'player'
    form_fields = ['quiz3', 'quiz4', 'quiz5']
    # This is for comprehension check
    @staticmethod
    def error_message(player: Player, values):
        solutions = dict(quiz3= 'Yes', quiz4= 'Yes', quiz5= 'No')
        error_messages = {
            'quiz3': 'You should have answered “Yes” to this question. If you could choose between receiving the outcomes of the random decision tree and £5 or less, you would prefer the outcomes of the random decision tree. (For an amount of £11 or more, you would prefer the fixed amount of money if you made the selection as in the picture.)',
            'quiz4':'You should have answered “Yes”. That is exactly what this graph means.',
            'quiz5':'You should have answered “No”. There is no path from the loss of £10 in the period in three days to the gain of £7 in the period in six days.'
        }
        errors = {
            name: error_messages.get(name, 'Incorrect answer. Please try again.')
            for name in solutions
            if values[name] != solutions[name]
        }
        if errors:
            player.num_failed_attempts += 1
            if player.num_failed_attempts >= 5:
                player.failed_too_many_1 = True
            else:
                return errors
    
    @staticmethod
    def is_displayed(player):
        return player.round_number == 1
    
class Check3(Page):
    allow_back_button = True
    form_model = 'player'
    form_fields = ['quiz1']
    # This is for comprehension check
    @staticmethod
    def error_message(player: Player, values):
        solutions = dict(quiz1= 'Ottawa')
        errors = {name : "Incorrect answer. Please try again." for name in solutions.keys()}
        if errors:
            player.num_failed_attempts += 1
            if player.num_failed_attempts >= 5:
                player.failed_too_many_3 = True
            else:
                return errors
    
    @staticmethod
    def is_displayed(player):
        return player.round_number == 1

class Failed(Page):
    # This is for failed comprehension check

    @staticmethod
    def is_displayed(player: Player):
        return player.failed_too_many_1 or player.failed_too_many_2 or player.failed_too_many_3

class Draw(Page):
    # This page is only displayed in the final round to draw the lottery for payment.
    @staticmethod
    def is_displayed(player):
        return player.round_number == 15

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

class RevisionBeforeDraw(Page):
    # This page is displayed after welcome session but before the spinning of the lottery to show the selected lottery again.
    template_name = 'session1/Draw.html'
    @staticmethod
    def is_displayed(player):
        return player.round_number in Constants.continuation_rounds

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
            'is_revision': True,
            'realized_summary': realized_summary,
            'realized_nodes_json': json.dumps(realized_nodes),
            'final_payoff_text': final_payoff_text,
            'current_session_number': None,
        }

class RevisionBeforeDraw1(RevisionBeforeDraw):
    @staticmethod
    def is_displayed(player):
        return player.round_number == Constants.continuation_rounds[0]

    @staticmethod
    def vars_for_template(player):
        context = RevisionBeforeDraw.vars_for_template(player)
        context['current_session_number'] = 2
        context['realized_summary'] = []
        context['realized_nodes_json'] = json.dumps({})
        context['final_payoff_text'] = None
        return context


class RevisionBeforeDraw2(RevisionBeforeDraw):
    @staticmethod
    def is_displayed(player):
        return player.round_number == Constants.continuation_rounds[1]

    @staticmethod
    def vars_for_template(player):
        context = RevisionBeforeDraw.vars_for_template(player)
        context['current_session_number'] = 3
        context['realized_summary'] = []
        context['realized_nodes_json'] = json.dumps({})
        context['final_payoff_text'] = None
        return context

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
        choice_lottery = with_upcoming_payoff_range(lottery, start_period=2)
        fields = Play._choice_field_names(choice_lottery, base_offset=0)
        fields.append('cutoff_index')
        return fields

    @staticmethod
    def vars_for_template(player):
        store = ensure_payment_setup(player)
        ensure_realized_up_to(player, 1, store=store)
        conditional_lottery = get_conditional_lottery(player, realized_up_to=1)
        full_lottery = get_selected_lottery(player, store=store)
        choice_lottery = with_upcoming_payoff_range(conditional_lottery, start_period=2)
        context = build_play_context(player, choice_lottery=choice_lottery, display_lottery=full_lottery, base_offset=0)
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
        choice_lottery = with_upcoming_payoff_range(lottery, start_period=2)
        store_cutoff_choice(player, choice_lottery, base_offset=0)
        ensure_realized_up_to(player, 2, store=store)
        if player.round_number == Constants.continuation_rounds[0]:
            schedule_session_start(
                player,
                prefix='session3',
                wait_seconds=15,
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
        choice_lottery = with_upcoming_payoff_range(lottery, start_period=3)
        fields = Play._choice_field_names(choice_lottery, base_offset=0)
        fields.append('cutoff_index')
        return fields

    @staticmethod
    def vars_for_template(player):
        store = ensure_payment_setup(player)
        ensure_realized_up_to(player, 2, store=store)
        conditional_lottery = get_conditional_lottery(player, realized_up_to=2)
        full_lottery = get_selected_lottery(player, store=store)
        choice_lottery = with_upcoming_payoff_range(conditional_lottery, start_period=3)
        context = build_play_context(player, choice_lottery=choice_lottery, display_lottery=full_lottery, base_offset=0)
        realized_nodes = store.get('realized_nodes', {})
        accumulated_value = compute_realized_offset(realized_nodes, upto_period=2)
        accumulated = format_payoff_value(accumulated_value)
        context.update(
            continuation_stage=3,
            realized_period1=realized_nodes.get(1),
            realized_period2=realized_nodes.get(2),
            realized_period3=realized_nodes.get(3),
            selected_round=store.get('selected_round'),
            base_lottery_name=store.get('lottery_name'),
            realized_accumulated=accumulated,
        )
        base_name = store.get('lottery_name') or context['lottery_name']
        context['lottery_name'] = f"{base_name} – continuation after period 2"
        return context

    @staticmethod
    def before_next_page(player, timeout_happened):
        store = ensure_payment_setup(player)
        lottery = get_conditional_lottery(player, realized_up_to=2)
        choice_lottery = with_upcoming_payoff_range(lottery, start_period=3)
        store_cutoff_choice(player, choice_lottery, base_offset=0)
        ensure_realized_up_to(player, 3, store=store)
        final_payoff = compute_final_payoff(store)
        if final_payoff is not None:
            store['final_payoff'] = final_payoff
            realized_nodes = store.get('realized_nodes', {})
            final_node = realized_nodes.get(3)
            if final_node:
                store['final_outcome_label'] = final_node.get('label')
            player.participant.vars['session1_final_payoff'] = final_payoff


class RevisionSession2(Page):
    # This page is only displayed in the continuation rounds to show the drawn lottery.
    template_name = 'session1/Draw.html'
    @staticmethod
    def is_displayed(player):
        return player.round_number == 16

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

class RevisionSession3(Page):
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
        return player.round_number == Constants.continuation_rounds[0] #Constants.initial_evaluation_rounds

    @staticmethod
    def vars_for_template(player):
        return build_wheel_context(
            player,
            period=1,
            title='Spin for Session 2',
            description='We now realize the first period of your lottery. The outcome applies to Session 2 (Play2).',
            stage_label='Period 1 outcome',
            status_text='Click the button to reveal the branch you will face today.',
            button_label='Spin for Session 2',
            next_step='This outcome determines the branch you will evaluate today.',
        )


class WheelSession3(Page):
    """Fortune wheel to realize the second continuation outcome (for Session 3)."""
    template_name = 'session1/fortune_wheel.html'

    @staticmethod
    def is_displayed(player):
        return player.round_number == 17

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

class Session2(Page):
    @staticmethod
    def is_displayed(player):
        return player.round_number == Constants.continuation_rounds[0]

class Session3(Page):
    template_name = 'session1/session2.html'
    @staticmethod
    def is_displayed(player):
        return player.round_number == Constants.continuation_rounds[1]

class RevisionSession1(Page):
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



page_sequence = [
    Welcome,
    Session1,
    Check1,
    Failed,
    Introduction2,
    Check2,
    Failed,
    Introduction3,
    Failed,
    Play,
    Draw,
    BridgeSession2,
    WelcomeSession2,
    RevisionBeforeDraw1,
    WheelSession2,
    RevisionSession1,
    Session2,
    Play2,
    BridgeSession3,
    WelcomeSession3,
    RevisionBeforeDraw2,
    WheelSession3,
    RevisionSession2,
    Session3,
    Play3,
    #Revision_session3,
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


def custom_export(players):
    """Export key experiment data with minimal columns."""
    choice_fields = [f'chf_{i}' for i in range(1, Constants.max_choice_count + 1)]
    yield [
        'session_code',
        'participant_code',
        'participant_id_in_session',
        'round_number',
        'lottery_id',
        'lottery_stake',
        'lottery_name',
        'selected_lottery_name',
        'selected_round',
        'selected_option',
        'selected_choice',
        'selected_amount',
        'cutoff_index',
        'cutoff_amount',
        'fine_cutoff_index',
        'fine_cutoff_amount',
        'fine_selected_choice',
        'fine_selected_amount',
        'num_failed_attempts',
        'failed_too_many_1',
        'failed_too_many_2',
        'failed_too_many_3',
        'quiz1',
        'quiz2',
        'quiz3',
        'quiz4',
        'quiz5',
        'participant_time_started_utc',
        'session2_start',
        'session2_start_readable',
        'session3_start',
        'session3_start_readable',
        'realized_period1_label',
        'realized_period1_probability',
        'realized_period1_abs_prob',
        'realized_period2_label',
        'realized_period2_probability',
        'realized_period2_abs_prob',
        'realized_period3_label',
        'realized_period3_probability',
        'realized_period3_abs_prob',
        'final_outcome_label',
        'final_payoff',
    ] + choice_fields

    ordered_players = sorted(
        players, key=lambda p: (p.session_id, p.participant_id, p.round_number)
    )

    for player in ordered_players:
        participant = player.participant
        store = participant.vars.get(PAYMENT_STORE_KEY) or {}
        realized_nodes = store.get('realized_nodes') or {}
        realized_1 = realized_nodes.get(1) or {}
        realized_2 = realized_nodes.get(2) or {}
        realized_3 = realized_nodes.get(3) or {}
        lottery_meta = Constants.lotteries.get(player.lottery_id) or {}
        lottery_name = lottery_meta.get('name')
        lottery_stake = lottery_meta.get('stake')
        selected_lottery_name = (
            participant.vars.get('selected_lottery_name') or store.get('lottery_name')
        )
        session2_start = get_player_field(player, 'session2_start')
        session2_start_readable = get_player_field(player, 'session2_start_readable')
        session3_start = get_player_field(player, 'session3_start')
        session3_start_readable = get_player_field(player, 'session3_start_readable')
        if session2_start is None:
            session2_start = participant.vars.get('session2_start')
        if session2_start_readable is None:
            session2_start_readable = participant.vars.get('session2_start_readable')
        if session3_start is None:
            session3_start = participant.vars.get('session3_start')
        if session3_start_readable is None:
            session3_start_readable = participant.vars.get('session3_start_readable')

        row = [
            player.session.code,
            participant.code,
            participant.id_in_session,
            player.round_number,
            player.lottery_id,
            lottery_stake,
            lottery_name,
            selected_lottery_name,
            store.get('selected_round'),
            get_player_field(player, 'selected_option'),
            get_player_field(player, 'selected_choice'),
            get_player_field(player, 'selected_amount'),
            get_player_field(player, 'cutoff_index'),
            get_player_field(player, 'cutoff_amount'),
            get_player_field(player, 'fine_cutoff_index'),
            get_player_field(player, 'fine_cutoff_amount'),
            get_player_field(player, 'fine_selected_choice'),
            get_player_field(player, 'fine_selected_amount'),
            get_player_field(player, 'num_failed_attempts'),
            get_player_field(player, 'failed_too_many_1'),
            get_player_field(player, 'failed_too_many_2'),
            get_player_field(player, 'failed_too_many_3'),
            get_player_field(player, 'quiz1'),
            get_player_field(player, 'quiz2'),
            get_player_field(player, 'quiz3'),
            get_player_field(player, 'quiz4'),
            get_player_field(player, 'quiz5'),
            participant.time_started_utc,
            session2_start,
            session2_start_readable,
            session3_start,
            session3_start_readable,
            realized_1.get('label'),
            realized_1.get('probability'),
            realized_1.get('abs_prob'),
            realized_2.get('label'),
            realized_2.get('probability'),
            realized_2.get('abs_prob'),
            realized_3.get('label'),
            realized_3.get('probability'),
            realized_3.get('abs_prob'),
            store.get('final_outcome_label'),
            store.get('final_payoff'),
        ]
        row.extend(get_player_field(player, field) for field in choice_fields)
        yield row
