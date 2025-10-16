from otree.api import *
import numpy as np
import random

doc = """
Compound lottery with three periods for player evaluation.
Players make decisions using a multiple choice list.
"""

class Constants(BaseConstants):
    name_in_url = 'compound_lottery_session1'
    players_per_group = None
    num_rounds = 15
    
    options = ['Safe option', 'Risky option']
    short_choice_count = 10
    long_choice_count = 20
    max_choice_count = long_choice_count
    
    # Dictionary of lottery structures
    lotteries = {
        'lottery_1': {
            'name': 'Four-Outcome Three-Period Lottery 0',
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
            'name': 'Six-Outcome Three-Period Lottery 0',
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


# Dynamically add the multiple choice list fields (supports up to long list length)
for i in range(1, Constants.max_choice_count + 1):
    setattr(
        Player,
        f'chf_{i}',
        models.StringField(choices=Constants.options, blank=True)
    )


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
    def _choice_amounts(lottery):
        count = Constants.short_choice_count if lottery['outcome_number'] == 4 else Constants.long_choice_count
        amounts = np.linspace(
            lottery['min_payoff'],
            lottery['max_payoff'],
            count,
            endpoint=True
        )
        return [int(round(a)) for a in amounts]

    @staticmethod
    def get_form_fields(player):
        lottery = Constants.lotteries[player.lottery_id]
        count = len(Play._choice_amounts(lottery))
        fields = [f'chf_{i}' for i in range(1, count + 1)]
        fields.append('cutoff_index')
        return fields

    @staticmethod
    def vars_for_template(player):
        import json
        lottery = Constants.lotteries[player.lottery_id]
        amounts = Play._choice_amounts(lottery)
        field_names = [f'chf_{i}' for i in range(1, len(amounts) + 1)]
        choice_rows = [
            {
                'index': idx,
                'amount': amount,
                'abs_amount': abs(amount),
                'field_name': field_name
            }
            for idx, (amount, field_name) in enumerate(zip(amounts, field_names))
        ]
        return {
            'lottery': json.dumps(lottery),  # Pass as JSON string for JS
            'lottery_name': lottery['name'],
            'outcome_number': lottery['outcome_number'],
            'lottery_description': lottery['description'],
            'max_payoff': lottery['max_payoff'],
            'min_payoff': lottery['min_payoff'],
            'choice_rows': choice_rows,
            'accept_label': Constants.options[0],
            'play_label': Constants.options[1],
            'period_1': lottery['periods'][1],
            'period_2': lottery['periods'][2],
            'period_3': lottery['periods'][3]
        }

    @staticmethod
    def before_next_page(player, timeout_happened):
        lottery = Constants.lotteries[player.lottery_id]
        amounts = Play._choice_amounts(lottery)
        count = len(amounts)
        if count == 0:
            player.selected_choice = None
            player.selected_amount = None
            player.selected_option = None
            player.cutoff_amount = None
            return

        # Record the amount associated with the participant's cutoff choice
        if player.cutoff_index is not None:
            idx = max(0, min(count - 1, player.cutoff_index))
            player.cutoff_amount = amounts[idx]
            player.selected_choice = idx + 1
            player.selected_amount = amounts[idx]
            field_name = f'chf_{idx + 1}'
            player.selected_option = getattr(player, field_name)
        else:
            player.cutoff_amount = None
            player.selected_choice = None
            player.selected_amount = None
            player.selected_option = None

class Results(Page):
    @staticmethod
    def vars_for_template(player):
        return {
            'payment_level': player.selected_amount,
            'selected_option': player.selected_option
        }


page_sequence = [Welcome, session1, Introduction, Play]


def creating_session(subsession: Subsession):
    """Assign lotteries by round so each round uses a different definition."""
    lottery_ids = list(Constants.lotteries.keys())
    if not lottery_ids:
        return
    idx = (subsession.round_number - 1) % len(lottery_ids)
    lottery_for_round = lottery_ids[idx]
    for player in subsession.get_players():
        player.lottery_id = lottery_for_round
