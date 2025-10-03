from otree.api import *
import random

doc = """
Compound lottery with three periods for player evaluation.
Players make decisions using a multiple choice list.
"""

class Constants(BaseConstants):
    name_in_url = 'compound_lottery_session3'
    players_per_group = None
    num_rounds = 1
    
    payments = ['CHF 1', 'CHF 2', 'CHF 3', 'CHF 4', 'CHF 5', 'CHF 6']
    options = ['Grab and go', 'Stay and play']
    
    # Dictionary of lottery structures
    lotteries = {
        'lottery_1': {
            'name': 'Simple Three-Period Lottery',
            'description': 'A basic compound lottery with three periods',
            'periods': {
                0: [
                    {'label': 'Start', 'probability': 1.0, 'abs_prob' : 1.0}
                ],
                1: [
                    {'label': '+$10', 'probability': 0.6, 'from': 'Start', 'abs_prob' : 0.6},
                    {'label': '-$10', 'probability': 0.4, 'from': 'Start', 'abs_prob' : 0.4}
                ],
                2: [
                    {'label': '+$7', 'probability': 0.7, 'from': '+$10', 'abs_prob' : 0.42},
                    {'label': '-$1', 'probability': 0.3, 'from': '+$10', 'abs_prob' : 0.18},
                    {'label': '+$7.4', 'probability': 0.4, 'from': '-$10', 'abs_prob' : 0.16},
                    {'label': '-$12', 'probability': 0.6, 'from': '-$10', 'abs_prob' : 0.24}
                ],
                3: [
                    {'label': '+$8', 'probability': 0.8, 'from': '+$7', 'parent': '+$10', 'abs_prob' : 0.336},
                    {'label': '+$0', 'probability': 0.2, 'from': '+$7', 'parent': '+$10', 'abs_prob' : 0.084},
                    {'label': '+$2', 'probability': 1, 'from': '-$1', 'parent': '+$10', 'abs_prob' : 0.18},
                    {'label': '+$5', 'probability': 0.2, 'from': '+$7.4', 'parent': '-$10', 'abs_prob' : 0.096},
                    {'label': '-$5', 'probability': 0.8, 'from': '+$7.4', 'parent': '-$10', 'abs_prob' : 0.064},
                    {'label': '+$3', 'probability': 0.5, 'from': '-$12', 'parent': '-$10', 'abs_prob' : 0.12},
                    {'label': '-$15', 'probability': 0.5, 'from': '-$12', 'parent': '-$10', 'abs_prob' : 0.12}
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
    # Use the default lottery
    lottery_id = models.StringField(initial=Constants.default_lottery)
    
    # Multiple choice list decisions
    chf_1 = models.StringField(
        choices=Constants.options,
        widget=widgets.RadioSelect
    )
    chf_2 = models.StringField(
        choices=Constants.options,
        widget=widgets.RadioSelect
    )
    chf_3 = models.StringField(
        choices=Constants.options,
        widget=widgets.RadioSelect
    )
    chf_4 = models.StringField(
        choices=Constants.options,
        widget=widgets.RadioSelect
    )
    chf_5 = models.StringField(
        choices=Constants.options,
        widget=widgets.RadioSelect
    )
    chf_6 = models.StringField(
        choices=Constants.options,
        widget=widgets.RadioSelect
    )
    
    # Store the choice selected for payment
    selected_choice = models.IntegerField()
    
    # Store which option was chosen for the selected choice
    selected_option = models.StringField()


# PAGES
class Welcome(Page):
    pass

class session3(Page):
    pass

class Introduction(Page):
    pass


class Play(Page):
    form_model = 'player'
    form_fields = ['chf_1', 'chf_2', 'chf_3', 'chf_4', 'chf_5', 'chf_6']
    
    @staticmethod
    def vars_for_template(player):
        lottery = Constants.lotteries[player.lottery_id]
        
        return {
            'lottery': lottery,
            'lottery_name': lottery['name'],
            'lottery_description': lottery['description'],
            'period_0': lottery['periods'][0],
            'period_1': lottery['periods'][1],
            'period_2': lottery['periods'][2],
            'period_3': lottery['periods'][3]
        }
    
    @staticmethod
    def before_next_page(player, timeout_happened):
        # Randomly select one choice for payment
        player.selected_choice = random.randint(1, 6)
        
        # Get the selected option for the chosen payment level
        choice_attr = f'chf_{player.selected_choice}'
        player.selected_option = getattr(player, choice_attr)




class Survey(Page):
    pass

class Results(Page):
    @staticmethod
    def vars_for_template(player):
        payment_level = Constants.payments[player.selected_choice - 1]
        
        return {
            'payment_level': payment_level,
            'selected_option': player.selected_option
        }


page_sequence = [Welcome, session3, Introduction, Play, Survey]