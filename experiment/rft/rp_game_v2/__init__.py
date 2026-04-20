from otree.api import *
import json

from .helpers import (
    AMOUNT_PATTERN,
    LOTTERIES_PATH,
    PAYMENT_STORE_KEY,
    rng,
    _assign_lottery_node_ids,
    _is_low_stake,
    _lottery_outcome_count,
    _sample_period_node,
    _weighted_choice,
    _wheel_node_identifier,
    build_bridge_context,
    build_conditional_lottery,
    build_lottery_order,
    build_play_context,
    build_realized_display,
    build_refined_amounts,
    build_refinement_series,
    build_wheel_context,
    build_wheel_segments,
    choice_amounts,
    choice_field_names,
    choice_rows,
    clear_fine_cutoff,
    continuation_time_left,
    continuation_window_expired,
    compute_final_payoff,
    compute_realized_offset,
    compute_upcoming_payoff_range,
    cutoff_validation_errors,
    determine_lottery_blank,
    ensure_payment_lottery_selected,
    ensure_payment_setup,
    ensure_realized_up_to,
    format_payoff_value,
    get_conditional_lottery,
    get_participant_lottery_order,
    get_payment_store,
    get_player_field,
    get_selected_lottery,
    get_session_start_info,
    is_high_stake,
    load_calibration_lotteries,
    load_lotteries,
    load_test_lotteries,
    lottery_last_period,
    parse_payoff_label,
    persist_payment_store,
    populate_fine_cutoff,
    should_show_bridge,
    store_cutoff_choice,
    verify_turnstile_token,
    with_upcoming_payoff_range,
)

doc = """
Compound lottery with three periods for player evaluation.
Players make decisions using a multiple choice list.
"""


# MODELS
class Constants(BaseConstants):
    name_in_url = 'rp_game_v2'
    players_per_group = None
    num_rounds = 18
    calibration_rounds = 8
    initial_evaluation_rounds = 16
    paying_round_numbers = tuple(range(calibration_rounds + 1, initial_evaluation_rounds + 1))
    main_evaluation_rounds = len(paying_round_numbers)
    continuation_rounds = (17, 18)
    
    
    options = ['Safe option', 'Risky option']
    short_choice_count_first = 20
    short_choice_count_later = 20
    long_choice_count = 20
    max_choice_count = long_choice_count
    high_refinement_count = 5
    
    # Dictionary of lottery structures (loaded from external JSON)
    lotteries = load_lotteries()
    calibration_lotteries = load_calibration_lotteries()
    evaluation_lotteries = lotteries
    practice_lotteries = load_test_lotteries()
    practice_lottery_id = next(iter(practice_lotteries), None)
    practice_lottery = (
        practice_lotteries[practice_lottery_id]
        if practice_lottery_id
        else None
    )

    # Default lottery to use
    default_lottery = 'lottery_1'


class Subsession(BaseSubsession):
    pass


class Group(BaseGroup):
    pass


class Player(BasePlayer):
    treatment_group = models.BooleanField(blank=True)
    demo_treatment_assignment = models.StringField(
        choices=[
            ['treatment', 'Treatment group'],
            ['control', 'Control group'],
        ],
        label='Which group should this participant be assigned to?',
        widget=widgets.RadioSelect,
    )
    demo_eligible_lottery_assignment = models.StringField(
        choices=[
            ['eligible', 'Receive an eligible payoff tree'],
            ['ineligible', 'Receive a non-eligible payoff tree'],
        ],
        label='Which payment-tree draw should this participant receive?',
        widget=widgets.RadioSelect,
    )
    num_failed_attempts = models.IntegerField(initial=0)
    failed_too_many_1 = models.BooleanField(initial=False)
    failed_too_many_2 = models.BooleanField(initial=False)
    failed_too_many_3 = models.BooleanField(initial=False)
    turnstile_token = models.StringField(blank=True)
    quiz1 = models.StringField(
        label='Do you need to be able to participate in all three sessions to participate in this study?',
        choices=['Yes', 'No'],
    )
    quiz2 = models.StringField(
        label='Is it fine to use AI in the study?',
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
        label='Please consider the following random payoff tree. Imagine that the random determination of the first outcome (three days from now) yields a loss of £10, as indicated by the red arrow to “-£10”. Is it possible that the outcome six days from now yields a gain of £7? ',
        choices=['Yes', 'No'],
    )
    quiz6 = models.StringField(
        label='During the decision tasks we asked you to evaluate random payoff trees. How difficult was it for you to understand your task? ',
        choices=['Not complicated at all', 'A bit complicated', 'Complicated', 'Very complicated', 'Too complicated or extremely complicated'],
    )
    quiz7 = models.StringField(
        label='Which of the following best describes your attention during the study?',
        choices=['I paid attention throughout the entire study.', 
                 'I paid attention throughout almost the entire study.', 
                 'I paid attention throughout most of the study.', 
                 'I paid attention  throughout parts of the study.', 
                 'I did not pay any attention at all during the study.' ],
    )
    quiz8 = models.StringField(
        label='3.	Please indicate your highest obtained educational qualification:',
        choices=['No formal qualification', 'GCSEs or equivalent (e.g., IGCSE, BTEC Firsts)', 'A-levels or equivalent (e.g., BTEC Nationals, T-levels)', 'Bachelor\'s degree or equivalent (e.g., BA, BSc, HND, HNC, foundation degree)', 'Master\'s degree or equivalent (e.g., MA, MSc, MRes, MBA)', 'Doctorate or equivalent (e.g., PhD, DPhil, EdD)'],
    )

    # Lottery assigned per round in creating_session
    lottery_id = models.StringField(blank=True)


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


CONTINUATION_MIN_SECONDS = 3


def _is_calibration_round_number(round_number):
    return 1 <= round_number <= Constants.calibration_rounds


def _is_main_evaluation_round_number(round_number):
    return round_number in Constants.paying_round_numbers


def _session1_round_context():
    return {
        'session1_rounds': Constants.initial_evaluation_rounds,
        'calibration_rounds': Constants.calibration_rounds,
        'evaluation_rounds': Constants.main_evaluation_rounds,
    }


def _session1_lotteries_for_round(round_number):
    if _is_calibration_round_number(round_number):
        return Constants.calibration_lotteries
    if _is_main_evaluation_round_number(round_number):
        return Constants.evaluation_lotteries
    return Constants.lotteries


def _get_round_lottery(player):
    lotteries = _session1_lotteries_for_round(player.round_number)
    lottery_id = player.lottery_id or Constants.default_lottery
    lottery = lotteries.get(lottery_id)
    if lottery is not None:
        return lottery
    return Constants.lotteries.get(lottery_id, Constants.lotteries[Constants.default_lottery])


def _continuation_time_left(player, prefix):
    return continuation_time_left(player, prefix)


def _continuation_has_time(player, prefix, min_seconds=CONTINUATION_MIN_SECONDS):
    seconds_left = _continuation_time_left(player, prefix)
    if seconds_left is None:
        return True
    return seconds_left > min_seconds


def _is_treatment_group(player):
    return bool(player.participant.vars.get('treatment_group', False))


def _assignment_selector_enabled(player):
    return (
        player.round_number == 1
        and bool(
            player.session.config.get(
                'assignment_selector',
                player.session.config.get('demo_assignment_selector', False),
            )
        )
    )


def _is_treatment_eligible(player):
    if not _is_treatment_group(player):
        return False
    store = ensure_payment_setup(player)
    lottery_id = store.get('lottery_id')
    if not lottery_id:
        return False
    lottery = Constants.lotteries.get(lottery_id, {})
    return lottery.get('description') == 'treatment'


def _int_or_none(value):
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


def _number_or_none(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _player_in_round_or_none(player, round_number):
    round_int = _int_or_none(round_number)
    if round_int is None or round_int < 1 or round_int > Constants.num_rounds:
        return None
    return player.in_round(round_int)


def _cutoff_values_from_player(source_player, first_accept_fallback=None):
    if source_player is None:
        return None, first_accept_fallback

    first_accept_value = get_player_field(source_player, 'fine_selected_amount')
    if first_accept_value is None:
        first_accept_value = get_player_field(source_player, 'selected_amount')
    if first_accept_value is None:
        first_accept_value = first_accept_fallback

    last_stay_value = get_player_field(source_player, 'fine_cutoff_amount')
    if last_stay_value is None:
        cutoff_index = _int_or_none(get_player_field(source_player, 'cutoff_index'))
        last_stay_value = None if cutoff_index == 0 else get_player_field(source_player, 'cutoff_amount')

    return last_stay_value, first_accept_value


def _treatment_cutoff_values(player, store, tree_version):
    if tree_version == 'session1':
        source_player = _player_in_round_or_none(player, store.get('selected_round'))
        return _cutoff_values_from_player(source_player, store.get('selected_amount'))
    if tree_version == 'session2':
        source_player = _player_in_round_or_none(player, Constants.continuation_rounds[0])
        return _cutoff_values_from_player(source_player, store.get('selected_amount_s2'))
    if tree_version == 'session3':
        return _cutoff_values_from_player(player, store.get('selected_amount_s3'))
    return None, None


def _offer_accepted(offer, first_accept_value):
    offer_number = _number_or_none(offer)
    first_accept_number = _number_or_none(first_accept_value)
    return (
        offer_number is not None
        and first_accept_number is not None
        and offer_number >= first_accept_number
    )


# Dynamically add the multiple choice list fields (supports up to long list length)
for i in range(1, Constants.max_choice_count + 1):
    setattr(
        Player,
        f'chf_{i}',
        models.StringField(choices=Constants.options, blank=True),
    )




# PAGES
class DemoAssignment(Page):
    form_model = 'player'
    form_fields = ['demo_treatment_assignment', 'demo_eligible_lottery_assignment']

    @staticmethod
    def is_displayed(player):
        return _assignment_selector_enabled(player)

    @staticmethod
    def vars_for_template(player):
        eligible_count = sum(
            1
            for lottery in Constants.evaluation_lotteries.values()
            if lottery.get('description') == 'treatment'
        )
        ineligible_count = max(0, len(Constants.evaluation_lotteries) - eligible_count)
        return {
            'eligible_lottery_count': eligible_count,
            'ineligible_lottery_count': ineligible_count,
        }

    @staticmethod
    def before_next_page(player, timeout_happened):
        treatment_group = player.demo_treatment_assignment == 'treatment'
        force_eligible_lottery = player.demo_eligible_lottery_assignment == 'eligible'

        player.participant.vars['treatment_group'] = treatment_group
        player.participant.vars['force_eligible_lottery'] = force_eligible_lottery
        player.participant.vars.pop('demo_force_eligible_lottery', None)

        for round_player in player.in_all_rounds():
            round_player.treatment_group = treatment_group

        player.participant.vars.pop(PAYMENT_STORE_KEY, None)
        player.participant.vars.pop('selected_lottery_name', None)
        player.participant.vars.pop('treatment_store', None)


class Welcome(Page):
    form_model = 'player'
    form_fields = ['turnstile_token']

    @staticmethod
    def is_displayed(player):
        return player.round_number == 1
    
    @staticmethod
    def vars_for_template(player):
        return {
            'is_treatment_group': player.participant.vars.get('treatment_group', False),
        }

    @staticmethod
    def error_message(player, values):
        token = values.get('turnstile_token')
        ok, _ = verify_turnstile_token(token)
        if not ok:
            return 'Please complete the verification to proceed.'
    pass

class Session1(Page):
    template_name = 'rp_game_v2/session1.html'
    allow_back_button = True
    @staticmethod
    def is_displayed(player):
        return player.round_number == 1
    pass

class Introduction1(Page):
    allow_back_button = True

    @staticmethod
    def is_displayed(player):
        return player.round_number == 1

    @staticmethod
    def vars_for_template(player):
        ctx = _session1_round_context()
        ctx['is_treatment_group'] = player.participant.vars.get('treatment_group', False)
        return ctx


class Introduction2(Page):
    allow_back_button = True

    @staticmethod
    def is_displayed(player):
        return player.round_number == 1

    @staticmethod
    def vars_for_template(player):
        ctx = _session1_round_context()
        ctx['is_treatment_group'] = player.participant.vars.get('treatment_group', False)
        return ctx


class Introduction3(Page):
    allow_back_button = True

    @staticmethod
    def is_displayed(player):
        return player.round_number == 1

    @staticmethod
    def vars_for_template(player):
        return {
            'is_treatment_group': player.participant.vars.get('treatment_group', False),
        }


class TreatmentInstructions(Page):
    allow_back_button = True

    @staticmethod
    def is_displayed(player):
        return player.round_number == 1 and player.participant.vars.get('treatment_group', False)


class LotterySet1(Page):
    @staticmethod
    def is_displayed(player):
        return player.round_number == 1
    pass

class LotterySet2(Page):
    @staticmethod
    def is_displayed(player):
        return player.round_number == Constants.calibration_rounds + 1
    pass

    @staticmethod
    def vars_for_template(player):
        return {
            'is_treatment_group': player.participant.vars.get('treatment_group', False),
        }


class PracticePlay(Page):
    form_model = 'player'

    @staticmethod
    def is_displayed(player):
        return player.round_number == 1

    @staticmethod
    def _choice_amounts(player, lottery, base_offset=0):
        return choice_amounts(player, lottery, base_offset)

    @staticmethod
    def _choice_field_names(player, lottery, base_offset=0):
        return choice_field_names(player, lottery, base_offset)

    @staticmethod
    def _choice_rows(player, lottery, base_offset=0):
        return choice_rows(player, lottery, base_offset)

    @staticmethod
    def get_form_fields(player):
        lottery = Constants.practice_lottery
        choice_lottery = with_upcoming_payoff_range(lottery, start_period=1)
        fields = Play._choice_field_names(player, choice_lottery, base_offset=0)
        fields.append('cutoff_index')
        if is_high_stake(lottery):
            fields.append('fine_cutoff_index')
        return fields

    @staticmethod
    def error_message(player, values):
        lottery = Constants.practice_lottery
        return cutoff_validation_errors(values, lottery)

    @staticmethod
    def vars_for_template(player):
        lottery = Constants.practice_lottery
        choice_lottery = with_upcoming_payoff_range(lottery, start_period=1)
        return build_play_context(player, choice_lottery=choice_lottery, display_lottery=lottery, base_offset=0)

    @staticmethod
    def before_next_page(player, timeout_happened):
        # Do not persist practice responses into the real round.
        for i in range(1, Constants.max_choice_count + 1):
            setattr(player, f'chf_{i}', None)
        player.cutoff_index = None
        player.cutoff_amount = None
        player.selected_choice = None
        player.selected_amount = None
        player.selected_option = None
        clear_fine_cutoff(player)

class Play(Page):
    form_model = 'player'



    @staticmethod
    def is_displayed(player):
        return False

    @staticmethod
    def _choice_amounts(player, lottery, base_offset=0):
        return choice_amounts(player, lottery, base_offset)

    @staticmethod
    def _choice_field_names(player, lottery, base_offset=0):
        return choice_field_names(player, lottery, base_offset)

    @staticmethod
    def _choice_rows(player, lottery, base_offset=0):
        return choice_rows(player, lottery, base_offset)

    @staticmethod
    def get_form_fields(player):
        lottery = _get_round_lottery(player)
        choice_lottery = with_upcoming_payoff_range(lottery, start_period=1)
        fields = Play._choice_field_names(player, choice_lottery, base_offset=0)
        fields.append('cutoff_index')
        if is_high_stake(lottery):
            fields.append('fine_cutoff_index')
        return fields

    @staticmethod
    def error_message(player, values):
        lottery = _get_round_lottery(player)
        return cutoff_validation_errors(values, lottery)

    @staticmethod
    def vars_for_template(player):
        lottery = _get_round_lottery(player)
        choice_lottery = with_upcoming_payoff_range(lottery, start_period=1)
        return build_play_context(player, choice_lottery=choice_lottery, display_lottery=lottery, base_offset=0)

    @staticmethod
    def before_next_page(player, timeout_happened):
        lottery = _get_round_lottery(player)
        choice_lottery = with_upcoming_payoff_range(lottery, start_period=1)
        store_cutoff_choice(player, choice_lottery)
        if player.round_number == Constants.initial_evaluation_rounds:
            ensure_payment_lottery_selected(player)


class Play_ez(Page):
    form_model = 'player'



    @staticmethod
    def is_displayed(player):
        return player.round_number in range(1, Constants.initial_evaluation_rounds + 1)

    @staticmethod
    def _choice_amounts(player, lottery, base_offset=0):
        return choice_amounts(player, lottery, base_offset)

    @staticmethod
    def _choice_field_names(player, lottery, base_offset=0):
        return choice_field_names(player, lottery, base_offset)

    @staticmethod
    def _choice_rows(player, lottery, base_offset=0):
        return choice_rows(player, lottery, base_offset)

    @staticmethod
    def get_form_fields(player):
        lottery = _get_round_lottery(player)
        choice_lottery = with_upcoming_payoff_range(lottery, start_period=1)
        fields = Play._choice_field_names(player, choice_lottery, base_offset=0)
        fields.append('cutoff_index')
        if is_high_stake(lottery):
            fields.append('fine_cutoff_index')
        return fields

    @staticmethod
    def error_message(player, values):
        lottery = _get_round_lottery(player)
        return cutoff_validation_errors(values, lottery)

    @staticmethod
    def vars_for_template(player):
        lottery = _get_round_lottery(player)
        choice_lottery = with_upcoming_payoff_range(lottery, start_period=1)
        return build_play_context(player, choice_lottery=choice_lottery, display_lottery=lottery, base_offset=0)

    @staticmethod
    def before_next_page(player, timeout_happened):
        lottery = _get_round_lottery(player)
        choice_lottery = with_upcoming_payoff_range(lottery, start_period=1)
        store_cutoff_choice(player, choice_lottery)
        if player.round_number == Constants.initial_evaluation_rounds:
            ensure_payment_lottery_selected(player)



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
            'quiz1': 'You should have chosen “Yes” for this question. Please only participate in this study if you can take part in all three sessions. Sessions two and three take place three and six days from now and will be much shorter than today’s session (only about 7-15 minutes each).',
            'quiz2': 'You should have chosen “No” for this question.  We are interested in your personal preferences (there are no right or wrong answers). Therefore, please do not use AI during the study.',
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

class Failed(Page):
    # This is for failed comprehension check

    @staticmethod
    def is_displayed(player: Player):
        if player.round_number in Constants.continuation_rounds:
            prefix = 'session2' if player.round_number == Constants.continuation_rounds[0] else 'session3'
            if continuation_window_expired(player, prefix, min_seconds=CONTINUATION_MIN_SECONDS):
                return True
        return player.failed_too_many_1 or player.failed_too_many_2 or player.failed_too_many_3

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
        is_treatment = player.participant.vars.get('treatment_group', False)
        is_eligible = (
            is_treatment
            and Constants.lotteries.get(store.get('lottery_id', ''), {}).get('description') == 'treatment'
        )
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
            'evaluation_rounds': Constants.main_evaluation_rounds,
            'is_treatment_group': is_treatment,
            'is_treatment_eligible': is_eligible,
        }

    @staticmethod
    def before_next_page(player, timeout_happened):
        ensure_payment_lottery_selected(player)

class RevisionBeforeDraw(Page):
    # This page is displayed after welcome session but before the spinning of the lottery to show the selected lottery again.
    template_name = 'rp_game_v2/Draw.html'
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
            'evaluation_rounds': Constants.main_evaluation_rounds,
        }

class RevisionBeforeDraw1(RevisionBeforeDraw):
    timer_text = 'Time left to complete this session:'

    @staticmethod
    def get_timeout_seconds(player):
        return _continuation_time_left(player, 'session2')

    @staticmethod
    def is_displayed(player):
        return player.round_number == Constants.continuation_rounds[0] and _continuation_has_time(player, 'session2')

    @staticmethod
    def vars_for_template(player):
        context = RevisionBeforeDraw.vars_for_template(player)
        context['current_session_number'] = 2
        context['realized_summary'] = []
        context['realized_nodes_json'] = json.dumps({})
        context['final_payoff_text'] = None
        return context


class RevisionBeforeDraw2(RevisionBeforeDraw):
    timer_text = 'Time left to complete this session:'

    @staticmethod
    def get_timeout_seconds(player):
        return _continuation_time_left(player, 'session3')

    @staticmethod
    def is_displayed(player):
        return player.round_number == Constants.continuation_rounds[1] and _continuation_has_time(player, 'session3')

    @staticmethod
    def vars_for_template(player):
        context = RevisionBeforeDraw.vars_for_template(player)
        context['current_session_number'] = 3
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


class Session2TimedPage(Page):
    timer_text = 'Time left to complete this session:'

    @staticmethod
    def get_timeout_seconds(player):
        return _continuation_time_left(player, 'session2')


class Session3TimedPage(Page):
    timer_text = 'Time left to complete this session:'

    @staticmethod
    def get_timeout_seconds(player):
        return _continuation_time_left(player, 'session3')


class Play2(Session2TimedPage):
    form_model = 'player'

    @staticmethod
    def is_displayed(player):
        return player.round_number == Constants.continuation_rounds[0] and _continuation_has_time(player, 'session2')

    @staticmethod
    def get_form_fields(player):
        store = ensure_payment_setup(player)
        ensure_realized_up_to(player, 1, store=store)
        lottery = get_conditional_lottery(player, realized_up_to=1)
        choice_lottery = with_upcoming_payoff_range(lottery, start_period=2)
        fields = Play._choice_field_names(player, choice_lottery, base_offset=0)
        fields.append('cutoff_index')
        if is_high_stake(lottery):
            fields.append('fine_cutoff_index')
        return fields

    @staticmethod
    def error_message(player, values):
        store = ensure_payment_setup(player)
        ensure_realized_up_to(player, 1, store=store)
        lottery = get_conditional_lottery(player, realized_up_to=1)
        return cutoff_validation_errors(values, lottery)

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
        store['selected_amount_s2'] = get_player_field(player, 'selected_amount')
        store['cutoff_index_s2'] = get_player_field(player, 'cutoff_index')
        store['cutoff_amount_s2'] = get_player_field(player, 'cutoff_amount')
        store['fine_selected_amount_s2'] = get_player_field(player, 'fine_selected_amount')
        store['fine_cutoff_amount_s2'] = get_player_field(player, 'fine_cutoff_amount')
        persist_payment_store(player, store)

class Play3(Session3TimedPage):
    form_model = 'player'
    template_name = 'rp_game_v2/Play3.html'

    @staticmethod
    def is_displayed(player):
        return player.round_number == Constants.continuation_rounds[1] and _continuation_has_time(player, 'session3')

    @staticmethod
    def get_form_fields(player):
        store = ensure_payment_setup(player)
        ensure_realized_up_to(player, 2, store=store)
        lottery = get_conditional_lottery(player, realized_up_to=2)
        choice_lottery = with_upcoming_payoff_range(lottery, start_period=3)
        fields = Play._choice_field_names(player, choice_lottery, base_offset=0)
        fields.append('cutoff_index')
        if is_high_stake(lottery):
            fields.append('fine_cutoff_index')
        return fields

    @staticmethod
    def error_message(player, values):
        store = ensure_payment_setup(player)
        ensure_realized_up_to(player, 2, store=store)
        lottery = get_conditional_lottery(player, realized_up_to=2)
        return cutoff_validation_errors(values, lottery)

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
        store['selected_amount_s3'] = get_player_field(player, 'selected_amount')
        store['cutoff_index_s3'] = get_player_field(player, 'cutoff_index')
        store['cutoff_amount_s3'] = get_player_field(player, 'cutoff_amount')
        store['fine_selected_amount_s3'] = get_player_field(player, 'fine_selected_amount')
        store['fine_cutoff_amount_s3'] = get_player_field(player, 'fine_cutoff_amount')
        ensure_realized_up_to(player, 3, store=store)
        final_payoff = compute_final_payoff(store)
        if final_payoff is not None:
            store['final_payoff'] = final_payoff
            realized_nodes = store.get('realized_nodes', {})
            final_period = lottery_last_period(get_selected_lottery(player, store=store))
            final_node = realized_nodes.get(final_period) if final_period is not None else None
            if final_node:
                store['final_outcome_label'] = final_node.get('label')
            player.participant.vars['session1_final_payoff'] = final_payoff
        persist_payment_store(player, store)


class RevisionSession2(Session2TimedPage):
    # This page is only displayed in the continuation rounds to show the drawn lottery.
    template_name = 'rp_game_v2/Draw.html'
    @staticmethod
    def is_displayed(player):
        return player.round_number == Constants.continuation_rounds[0] and _continuation_has_time(player, 'session2')

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
            'evaluation_rounds': Constants.main_evaluation_rounds,
        }

class RevisionSession3(Session3TimedPage):
    # This page is only displayed in the continuation rounds to show the drawn lottery.
    template_name = 'rp_game_v2/Draw.html'
    @staticmethod
    def is_displayed(player):
        return player.round_number == Constants.continuation_rounds[1] and _continuation_has_time(player, 'session3')

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
            'evaluation_rounds': Constants.main_evaluation_rounds,
        }

class WheelSession2(Session2TimedPage):
    """Fortune wheel to realize the first continuation outcome (for Session 2)."""
    template_name = 'rp_game_v2/fortune_wheel.html'

    @staticmethod
    def is_displayed(player):
        return player.round_number == Constants.continuation_rounds[0] and _continuation_has_time(player, 'session2')

    @staticmethod
    def vars_for_template(player):
        return build_wheel_context(
            player,
            period=1,
            title='Determination of the first outcome',
            description='Now the first outcome of your random payoff tree will be determined.',
            stage_label='Outcome determination (first outcome, Session 2)',
            status_text='Click the button to spin the wheel of fortune determining the first outcome.',
            button_label='Click here to spin the wheel',
            next_step='This outcome determines the branch you will evaluate today.',
        )


class WheelSession3(Session3TimedPage):
    """Fortune wheel to realize the second continuation outcome (for Session 3)."""
    template_name = 'rp_game_v2/fortune_wheel.html'

    @staticmethod
    def is_displayed(player):
        return player.round_number == Constants.continuation_rounds[1] and _continuation_has_time(player, 'session3')

    @staticmethod
    def vars_for_template(player):
        return build_wheel_context(
            player,
            period=2,
            title='Determination of the second outcome',
            description='Now the second outcome of your random payoff tree will be determined.',
            stage_label='Outcome determination (second outcome, Session 3)',
            status_text='Click the button to spin the wheel of fortune determining the second outcome.',
            button_label='Click here to spin the wheel',
            next_step='This outcome determines the branch you will evaluate today.',
        )



class TreatmentPayoff(Session3TimedPage):
    """Die roll + monetary offer page for treatment-eligible participants (Session 3)."""

    @staticmethod
    def is_displayed(player):
        if player.round_number != Constants.continuation_rounds[1]:
            return False
        if not _continuation_has_time(player, 'session3'):
            return False
        return _is_treatment_eligible(player)

    @staticmethod
    def vars_for_template(player):
        store = ensure_payment_setup(player)
        t_store = player.participant.vars.setdefault('treatment_store', {})

        if 'die_roll' not in t_store:
            die_roll = rng.randint(1, 6)
            if die_roll <= 2:
                tree_version = 'session1'
                lottery_for_range = get_selected_lottery(player, store=store)
                start_period = 1
                reported_value = store.get('selected_amount')
            elif die_roll <= 4:
                tree_version = 'session2'
                lottery_for_range = get_conditional_lottery(player, realized_up_to=1)
                start_period = 2
                reported_value = store.get('selected_amount_s2')
            else:
                tree_version = 'session3'
                lottery_for_range = get_conditional_lottery(player, realized_up_to=2)
                start_period = 3
                reported_value = store.get('selected_amount_s3')

            min_outcome, max_outcome = compute_upcoming_payoff_range(lottery_for_range, start_period)
            if min_outcome is None or max_outcome is None:
                min_outcome = store.get('lottery_id') and Constants.lotteries.get(store.get('lottery_id', ''), {}).get('min_payoff', 0)
                max_outcome = store.get('lottery_id') and Constants.lotteries.get(store.get('lottery_id', ''), {}).get('max_payoff', 0)
                min_outcome = min_outcome or 0
                max_outcome = max_outcome or 0

            last_stay_value, first_accept_value = _treatment_cutoff_values(player, store, tree_version)
            if first_accept_value is None:
                first_accept_value = reported_value
            reported_value = first_accept_value

            offer = rng.randint(int(min_outcome), int(max_outcome)) if max_outcome > min_outcome else int(min_outcome)
            offer_accepted = _offer_accepted(offer, first_accept_value)

            realized_nodes = store.get('realized_nodes', {})
            if tree_version == 'session1':
                tree_realized_outcome = compute_realized_offset(realized_nodes)
            elif tree_version == 'session2':
                tree_realized_outcome = (
                    compute_realized_offset(realized_nodes) - compute_realized_offset(realized_nodes, upto_period=1)
                )
            else:
                node3 = realized_nodes.get(3, {})
                tree_realized_outcome = parse_payoff_label(node3.get('label'))

            realized_nodes = store.get('realized_nodes', {})
            # Build per-period payment schedule (days from today = Session 3)
            if tree_version == 'session1':
                payment_schedule = [
                    {'period': "Period 2", 'label': (realized_nodes.get(1) or {}).get('label', ''), 'days': 3},
                    {'period': "Period 3", 'label': (realized_nodes.get(2) or {}).get('label', ''), 'days': 6},
                    {'period': "After Period 3", 'label': (realized_nodes.get(3) or {}).get('label', ''), 'days': 9},
                ]
            elif tree_version == 'session2':
                payment_schedule = [
                    {'period': "Period 3", 'label': (realized_nodes.get(2) or {}).get('label', ''), 'days': 3},
                    {'period': "After Period 3", 'label': (realized_nodes.get(3) or {}).get('label', ''), 'days': 6},
                ]
            else:
                payment_schedule = [
                    {'period': "After Period 3", 'label': (realized_nodes.get(3) or {}).get('label', ''), 'days': 3},
                ]

            t_store.update(
                die_roll=die_roll,
                tree_version=tree_version,
                offer=offer,
                offer_min=int(min_outcome),
                offer_max=int(max_outcome),
                reported_value=reported_value,
                last_stay_value=last_stay_value,
                first_accept_value=first_accept_value,
                offer_accepted=offer_accepted,
                tree_realized_outcome=tree_realized_outcome,
                payment_schedule=payment_schedule,
            )

        last_stay_value, first_accept_value = _treatment_cutoff_values(player, store, t_store.get('tree_version'))
        if first_accept_value is not None:
            t_store['first_accept_value'] = first_accept_value
            t_store['reported_value'] = first_accept_value
            t_store['offer_accepted'] = _offer_accepted(t_store.get('offer'), first_accept_value)
        if last_stay_value is not None or 'last_stay_value' not in t_store:
            t_store['last_stay_value'] = last_stay_value

        # Always build visualization data fresh (not persisted in t_store)
        full_lottery = get_selected_lottery(player, store=store)
        realized_nodes = store.get('realized_nodes', {})

        # Filter realized nodes to only the periods used for payment
        tree_version = t_store.get('tree_version')
        if tree_version == 'session1':
            display_periods = {1, 2, 3}
        elif tree_version == 'session2':
            display_periods = {2, 3}
        else:
            display_periods = {3}
        display_realized = {p: n for p, n in realized_nodes.items() if p in display_periods}

        ctx = dict(t_store)
        ctx['selected_lottery'] = json.dumps(full_lottery)
        ctx['realized_nodes_json'] = json.dumps(display_realized)
        ctx['lottery_name'] = store.get('lottery_name', '')
        return ctx


class Post(Session3TimedPage):
    form_model = 'player'
    form_fields = ['quiz6', 'quiz7', 'quiz8']
    @staticmethod
    def is_displayed(player):
        return player.round_number == Constants.continuation_rounds[1] and _continuation_has_time(player, 'session3')

class Thankyou(Session3TimedPage):
    @staticmethod
    def is_displayed(player):
        return player.round_number == Constants.continuation_rounds[1] and _continuation_has_time(player, 'session3')

    @staticmethod
    def vars_for_template(player):
        t_store = player.participant.vars.get('treatment_store', {})
        return {
            'is_treatment_eligible': _is_treatment_eligible(player),
            'treatment_die_roll': t_store.get('die_roll'),
            'treatment_tree_version': t_store.get('tree_version'),
            'treatment_offer': t_store.get('offer'),
            'treatment_reported_value': t_store.get('reported_value'),
            'treatment_last_stay_value': t_store.get('last_stay_value'),
            'treatment_first_accept_value': t_store.get('first_accept_value'),
            'treatment_offer_accepted': t_store.get('offer_accepted'),
            'treatment_realized_outcome': t_store.get('tree_realized_outcome'),
        }


class WelcomeSession2(Session2TimedPage):
    form_model = 'player'
    form_fields = ['turnstile_token']
    @staticmethod
    def is_displayed(player):
        return player.round_number == Constants.continuation_rounds[0] and _continuation_has_time(player, 'session2')

    @staticmethod
    def error_message(player, values):
        token = values.get('turnstile_token')
        ok, _ = verify_turnstile_token(token)
        if not ok:
            return 'Please complete the verification to proceed.'
    
class WelcomeSession3(Session3TimedPage):
    form_model = 'player'
    form_fields = ['turnstile_token']
    @staticmethod
    def is_displayed(player):
        return player.round_number == Constants.continuation_rounds[1] and _continuation_has_time(player, 'session3')
    @staticmethod
    def error_message(player, values):
        token = values.get('turnstile_token')
        ok, _ = verify_turnstile_token(token)
        if not ok:
            return 'Please complete the verification to proceed.'

class Session2(Session2TimedPage):
    template_name = 'rp_game_v2/session2.html'
    @staticmethod
    def is_displayed(player):
        return player.round_number == Constants.continuation_rounds[0] and _continuation_has_time(player, 'session2')
    
    @staticmethod
    def vars_for_template(player):
        return {
            'is_treatment_group': player.participant.vars.get('treatment_group', False),
            'is_eligible_for_treatment': _is_treatment_eligible(player),
        }

class Session3(Session3TimedPage):
    template_name = 'rp_game_v2/session2.html'
    @staticmethod
    def is_displayed(player):
        return player.round_number == Constants.continuation_rounds[1] and _continuation_has_time(player, 'session3')

    @staticmethod
    def vars_for_template(player):
        return {
            'is_treatment_group': player.participant.vars.get('treatment_group', False),
            'is_eligible_for_treatment': _is_treatment_eligible(player),
        }
class RevisionSession1(Session2TimedPage):
    template_name = 'rp_game_v2/Draw.html'
    @staticmethod
    def is_displayed(player):
        return player.round_number == Constants.continuation_rounds[0] and _continuation_has_time(player, 'session2')
    
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
            'evaluation_rounds': Constants.main_evaluation_rounds,
        }


# ------------- Page Sequence Control -------------
page_sequence = [
    DemoAssignment,
    Welcome,
    Session1,
    Check1,
    Failed,
    Introduction2,
    Check2,
    Failed,
    Introduction3,
    TreatmentInstructions,
    Failed,
    PracticePlay,
    LotterySet1,
    LotterySet2,
    Play_ez,
    Play,
    Draw,
    WelcomeSession2,
    RevisionBeforeDraw1,
    WheelSession2,
    RevisionSession1,
    Session2,
    Play2,
    WelcomeSession3,
    RevisionBeforeDraw2,
    WheelSession3,
    RevisionSession3,
    Session3,
    Play3,
    TreatmentPayoff,
    Post,
    Thankyou,
    BridgeSession2,
    BridgeSession3
]


# SESSION HOOKS
def creating_session(subsession: Subsession):
    """Assign separate Session 1 lottery orders for calibration and evaluation."""
    if subsession.round_number == 1:
        for player in subsession.get_players():
            if 'treatment_group' not in player.participant.vars:
                player.participant.vars['treatment_group'] = (rng.random() < 0.5)
            player.treatment_group = player.participant.vars['treatment_group']

    for player in subsession.get_players():
        if _is_calibration_round_number(subsession.round_number):
            lottery_order = get_participant_lottery_order(
                player.participant,
                store_key='calibration_lottery_order',
                lotteries=Constants.calibration_lotteries,
            )
            if not lottery_order:
                continue
            phase_index = subsession.round_number - 1
            player.lottery_id = lottery_order[phase_index % len(lottery_order)]
            continue

        if _is_main_evaluation_round_number(subsession.round_number):
            lottery_order = get_participant_lottery_order(
                player.participant,
                store_key='evaluation_lottery_order',
                lotteries=Constants.evaluation_lotteries,
            )
            if not lottery_order:
                continue
            phase_index = subsession.round_number - Constants.calibration_rounds - 1
            player.lottery_id = lottery_order[phase_index % len(lottery_order)]


def custom_export(players):
    """Export key study data with minimal columns."""
    choice_fields = [f'chf_{i}' for i in range(1, Constants.max_choice_count + 1)]
    yield [
        'session_code',
        'participant_code',
        'participant_label',
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
        'quiz6',
        'quiz7',
        'quiz8',
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
        'treatment_group',
        'treatment_die_roll',
        'treatment_tree_version',
        'treatment_offer',
        'treatment_reported_value',
        'treatment_last_stay_value',
        'treatment_first_accept_value',
        'treatment_offer_accepted',
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
        lottery_source = _session1_lotteries_for_round(player.round_number)
        lottery_meta = lottery_source.get(player.lottery_id) or Constants.lotteries.get(player.lottery_id) or {}
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
            participant.label,
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
            get_player_field(player, 'quiz6'),
            get_player_field(player, 'quiz7'),
            get_player_field(player, 'quiz8'),
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
            participant.vars.get('treatment_group'),
            (participant.vars.get('treatment_store') or {}).get('die_roll'),
            (participant.vars.get('treatment_store') or {}).get('tree_version'),
            (participant.vars.get('treatment_store') or {}).get('offer'),
            (participant.vars.get('treatment_store') or {}).get('reported_value'),
            (participant.vars.get('treatment_store') or {}).get('last_stay_value'),
            (participant.vars.get('treatment_store') or {}).get('first_accept_value'),
            (participant.vars.get('treatment_store') or {}).get('offer_accepted'),
        ]
        row.extend(get_player_field(player, field) for field in choice_fields)
        yield row
