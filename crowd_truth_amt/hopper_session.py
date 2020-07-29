import urllib.parse
import urllib.request
import json
import getpass
from collections import defaultdict


class Session:
    def __init__(self, env_name="prod", token_loc='.api_session_token'):

        # Lifted from here https://github.com/avalanche-strategy/hopper/blob/master/hopper/scripts/get_token.py
        # Thank you, Mark!
        username = input("Username: ")
        password = getpass.getpass()

        base_urls = {
            "prod": "https://app.avalancheinsights.com/",
            "dev": "https://dev.avalancheinsights.com/",
            "local": "http://localhost:3000/",
        }

        self.base_url = base_urls[env_name]
        self.login_url = self.base_url + 'api/users/auth/login/'
        # data = urllib.parse.urlencode({"password": password, "username": username})
        #
        # req = urllib.request.Request(
        #     base_urls[env_name] + "api/users/auth/login/",
        #     data.encode("ascii"),
        #     method="POST",
        # )
        #
        # with urllib.request.urlopen(req) as response:
        #     the_page = response.read()

        self.token = "Token " + self.get_token()
        self.headers = {"authorization": self.token}

    def authorize_access():
        username = input("Username: ")
        password = getpass.getpass()

        data = urllib.parse.urlencode({"password": password, "username": username})

        req = urllib.request.Request(
            self.login_url,
            data.encode("ascii"),
            method="POST",
        )

        with urllib.request.urlopen(req) as response:
            the_page = response.read()
            token = json.loads(the_page.decode())['token']

        return token

    def get_token(token_loc=self.token_loc, overwrite=False):
        if overwrite:
            token = authorize_access()
            with open(token_loc, 'w') as fOut:
                fOut.write(token)
            return token
        try:
            with open(token_loc, 'r') as fIn:
                token = fIn.read()
                return token
        except FileNotFoundError as fnfe:
             get_token(token_loc, overwrite=True)

    def api_get_call(self, url):
        req = urllib.request.Request(url, headers=self.headers, method="GET",)

        with urllib.request.urlopen(req) as response:
            the_page = response.read()
        try:
            response_dict = json.loads(the_page.decode())
        except JSONDecodeError as json_err:
            print(the_page)
            raise json_err

        return response_dict

    def get_all_surveys(self):
        return self.api_get_call(self.base_url + "api/surveys/surveys/")

    def get_survey(self, survey_id):
        return self.api_get_call(
            self.base_url + f"api/surveys/nested_surveys/{survey_id}/"
        )

    def get_text_analysis(self, analysis_id):
        return self.api_get_call(
            self.base_url + f"api/analysis/nested_text_analyses/{analysis_id}/"
        )

    def list_all_surveys(self):
        surveys = self.get_all_surveys()

        for survey in surveys:
            print(f"ID: {survey['id']}, Name: {survey['name']}")

        return surveys

    def list_questions_for_survey(self, survey_id=None):

        if survey_id is None:
            survey_id = input(
                "Enter the survey ID for the survey you'd like to see the questions for: "
            )

        survey = self.get_survey(survey_id)

        for question in survey["questions"]:
            print(f"ID: {question['id']}, Name:{question['name']}")

        return survey["questions"]

    def list_analyses_for_survey(self, survey_id=None):

        if survey_id is None:
            survey_id = input(
                "Enter the survey ID for the survey you'd like to see the analyses for: "
            )

        survey = self.get_survey(survey_id)

        for analysis in survey["text_analyses"]:
            print(f"ID: {analysis['id']}, Name:{analysis['name']}")

        return survey["text_analyses"]

    def get_survey_keywords(self, survey_id):
        out = defaultdict(list)

        survey_data = self.get_survey(survey_id)

        questions = {
            question["id"]: question["name"] for question in survey_data["questions"]
        }

        for text_analysis in survey_data["text_analyses"]:
            for elem in [
                "created",
                "survey",
                "keyword_sets",
                "rollups",
                "rollup_schemes",
            ]:
                text_analysis.pop(elem)

        for text_analysis in survey_data["text_analyses"]:
            out[questions[text_analysis["question"]]].append(
                {key: text_analysis[key] for key in ["id", "name"]}
            )

        for text_analyses in out.values():
            for text_analysis in text_analyses:
                feature_data = self.get_text_analysis(text_analysis["id"])
                text_analysis["keywords"] = [
                    keyword["keyword"]
                    for keyword_set in feature_data["keyword_sets"]
                    for keyword in keyword_set["keywords"]
                ]

        out = {
            "survey_id": survey_data["id"],
            "name": survey_data["name"],
            "questions": out,
        }

        return out
