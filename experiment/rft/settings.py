from os import environ

SESSION_CONFIGS = [
    dict(
        name='rp_game',
        app_sequence=['rp_game'],
        num_demo_participants=3,
    )

]

# if you set a property in SESSION_CONFIG_DEFAULTS, it will be inherited by all configs
# in SESSION_CONFIGS, except those that explicitly override it.
# the session config can be accessed from methods in your apps as self.session.config,
# e.g. self.session.config['participation_fee']

SESSION_CONFIG_DEFAULTS = dict(
    real_world_currency_per_point=1.00, participation_fee=0.00, doc=""
)

PARTICIPANT_FIELDS = ['selected_lottery_name']
SESSION_FIELDS = []

# ISO-639 code
# for example: de, fr, ja, ko, zh-hans
LANGUAGE_CODE = 'en'

# e.g. EUR, GBP, CNY, JPY
REAL_WORLD_CURRENCY_CODE = 'CHF'
USE_POINTS = True

ADMIN_USERNAME = 'admin'
# for security, best to set admin password in an environment variable
ADMIN_PASSWORD = environ.get('OTREE_ADMIN_PASSWORD')

DEMO_PAGE_INTRO_HTML = """ """

OTREE_AUTH_LEVEL = 'STUDY'  # for adding participants from the admin interface

SECRET_KEY = '6053408606911'
OTREE_ADMIN_NAME = 'xyguo'
OTREE_ADMIN_PASSWORD = 'unisg26'


ROOMS = [
    dict(
        name='rp_game_room',
        display_name='RP Game Room',
    ),
]

