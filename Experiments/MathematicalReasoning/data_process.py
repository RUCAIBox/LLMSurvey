import re
import json

class DataProcessForMATH:
    def __init__(self, data_path, is_clean=False, num_examplar=5):
        self.num_examplar = num_examplar
        self.data_path = data_path
        with open(self.data_path, 'r') as fin:
            raw_examplars = json.load(fin)

        self.examplars = []
        for examplar in raw_examplars:
            # if examplar['knowledge_point'] != 'algebra':
            #     continue
            self.examplars.append(self.process_single_data(examplar))

    def process(
            self,
            data,
            num_del_examplar=0,
            pattern_demo="Problem: {}\nSolution: Let's think step by step. {} <ans> {} </ans>\n",
            pattern_test="Problem: {}\nSolution: Let's think step by step.",
    ):
        processed_data = self.process_single_data(data)

        prompt = ''
        num_example = self.num_examplar - num_del_examplar
        for i in range(num_example):
            problem = self.examplars[i]['problem']
            solution = self.examplars[i]['cot']
            answer = self.examplars[i]['answer']
            prompt = prompt + pattern_demo.format(problem, solution, answer)
        prompt = prompt + 'You should follow the examples above and answer the following problem. You should use <ans> and  </ans> to show your answer.\n'
        prompt = prompt + pattern_test.format(processed_data['problem'])

        return prompt, processed_data['answer']
    
    def process_retrieval(
            self,
            data,
            num_del_examplar,
            ret_examplar,
            pattern_demo="Problem: {}\nSolution: Let's think step by step. {} The answer is {}\n",
            pattern_test="Problem: {}\nSolution: Let's think step by step.",
    ):
        processed_data = self.process_single_data(data)

        prompt = ''
        num_example = self.num_examplar - num_del_examplar
        for i in range(num_example):
            examplar = self.process_single_data(ret_examplar[i])
            problem = examplar['problem']
            solution = examplar['cot']
            answer = examplar['answer']
            prompt = prompt + pattern_demo.format(problem, solution, answer)
        prompt = prompt + pattern_test.format(processed_data['problem'])

        return prompt, processed_data['answer']

    def process_classifier(
            self,
            data,
            num_del_examplar=0,
            pattern_demo="Problem: {}\nProblem type: {}\n",
            pattern_test="Problem: {}\nProblem type: ",
    ):
        processed_data = self.process_single_data(data)

        prompt = 'You should classify the problem into counting_and_probability, geometry, intermediate_algebra, prealgebra, precalculus, algebra, number_theory\n'
        num_example = self.num_examplar - num_del_examplar
        for i in range(num_example):
            problem = self.examplars[i]['problem']
            prob_type = self.examplars[i]['knowledge_point']
            prompt = prompt + pattern_demo.format(problem, prob_type)
        prompt = prompt + pattern_test.format(processed_data['problem'])

        return prompt, processed_data['knowledge_point']

    def get_answer(self, content):
        pattern = '\\boxed'
        start_pos = content.rfind(pattern)
        answer = ''
        num_left = 0
        for i in range(start_pos + 7, len(content)):
            if (content[i] == '}' and num_left == 0):
                break
            if (content[i] == '{'):
                num_left = num_left + 1
            elif (content[i] == '}'):
                num_left = num_left - 1
            answer = answer + content[i]
        return answer


    def process_single_data(self, data):
        processed_data = {
                'problem': data['problem'],
                'cot': data['solution'],
                'answer': self.get_answer(data['solution']),
        }
        if ('knowledge_point' in data):
            processed_data['knowledge_point'] = data['knowledge_point']
        return processed_data

class DataProcessForGSM8k:
    def __init__(self, data_path, is_clean=False, num_examplar=5):
        self.num_examplar = num_examplar
        self.is_clean = is_clean
        self.data_path = data_path
        with open(self.data_path, 'r') as fin:
            raw_examplars = json.load(fin)
        
        self.examplars = []
        for examplar in raw_examplars:
            self.examplars.append(self.process_single_data(examplar))

    def process(
            self, 
            data,
            num_del_examplar=0,
            pattern_demo="Problem: {}\nSolution: Let's think step by step. {} <ans> {} </ans>\n",
            pattern_test="Problem: {}\nSolution: Let's think step by step.",
    ):
        processed_data = self.process_single_data(data)
        
        prompt = ''
        num_example = self.num_examplar - num_del_examplar
        for i in range(num_example):
            problem = self.examplars[i]['problem']
            solution = self.examplars[i]['cot']
            answer = self.examplars[i]['answer']
            prompt = prompt + pattern_demo.format(problem, solution, answer)
        prompt = prompt + 'You should follow the examples above and answer the following problem. You should use <ans> and  </ans> to show your answer.\n'
        prompt = prompt + pattern_test.format(processed_data['problem'])

        return prompt, processed_data['answer']
    
    def process_decompose(
            self, 
            data,
            num_del_examplar=0,
            pattern_demo="Problem: {}\nSolution: {}\nSub-Problem: Let's decompose problem into sub-problem according to the solution. {}\n",
            pattern_test="Problem: {}\nSolution: {}\nSub-Problem: Let's decompose problem into sub-problem according to the solution.",
    ):
        processed_data = self.process_single_data(data)
        
        prompt = ''
        num_example = self.num_examplar - num_del_examplar
        for i in range(num_example):
            problem = self.examplars[i]['problem']
            solution = self.examplars[i]['cot']
            sub_prob = self.examplars[i]['sub-problem']
            prompt = prompt + pattern_demo.format(problem, solution, sub_prob)
        prompt = prompt + pattern_test.format(processed_data['problem'], processed_data['cot'])

        return prompt
    
    def process_decompose_wo_cot(
            self, 
            data,
            num_del_examplar=0,
            pattern_demo="Problem: {}\nSub-Problem: Let's decompose problem into sub-problem according to the solution. {}\n",
            pattern_test="Problem: {}\nSub-Problem: Let's decompose problem into sub-problem according to the solution.",
    ):
        processed_data = self.process_single_data(data)
        
        prompt = ''
        num_example = self.num_examplar - num_del_examplar
        for i in range(num_example):
            problem = self.examplars[i]['problem']
            sub_prob = self.examplars[i]['sub-problem']
            prompt = prompt + pattern_demo.format(problem, sub_prob)
        prompt = prompt + pattern_test.format(processed_data['problem'])

        return prompt

    def process_solve_decompose(
            self, 
            test_data=None, 
            num_del_examplar=0
    ):
        processed_data = self.process_single_data(test_data)

        problem_pattern = "Problem: {}\nLet's solve the problem by decomposing it into several sub-problem. {}According the analysis above, the answer is {}\n"
        sub_problem_pattern = "Sub-Problem {}: {}\nSolution {}: Let's think step by step. {}\n"
        prompt = ""
        num_examplar = self.num_examplar - num_del_examplar
        for i in range(num_examplar):
            data = self.examplars[i]
            prob_prompt = ""
            sub_prob_prompt = ""
            idx = 0
            for sub_prob, sub_prob_cot in zip(data['sub-problem'].split('\n'), data['sub-problem-cot'].split('\n')):
                idx = idx + 1
                sub_prob = sub_prob.split(f'{idx}.')[-1].strip()
                sub_prob_cot = sub_prob_cot.strip()
                sub_prob_prompt = sub_prob_prompt + sub_problem_pattern.format(idx, sub_prob, idx, sub_prob_cot)
            prob_prompt = problem_pattern.format(data['problem'], sub_prob_prompt, data['answer'])
            prompt = prompt + prob_prompt
        return prompt, processed_data
        

    def process_solve_decompose_together(
            self, 
            test_data=None, 
            num_del_examplar=0
    ):
        processed_data = self.process_single_data(test_data)

        problem_pattern = "Example Problem: {}\nExample Solution: Let’s break down this problem: {}\n{}The answer is {}\n"
        prompt = ""
        num_examplar = self.num_examplar - num_del_examplar
        for i in range(num_examplar):
            data = self.examplars[i]
            prob_prompt = ""
            sub_prob_prompt = ""
            sub_solu_prompt = ""
            idx = 0
            for sub_prob, sub_prob_cot in zip(data['sub-problem'].split('\n'), data['sub-problem-cot'].split('\n')):
                idx = idx + 1
                
                sub_prob = sub_prob.strip()
                sub_prob_prompt = sub_prob_prompt + sub_prob + ' '

                sub_prob_cot = sub_prob_cot.strip()
                sub_solu_prompt = sub_solu_prompt + '{}. {}\n'.format(idx, sub_prob_cot)

            prob_prompt = problem_pattern.format(data['problem'], sub_prob_prompt, sub_solu_prompt, data['answer'])
            prompt = prompt + prob_prompt
        
        test_problem_pattern = "Example Problem: {}\nExample Solution: Let’s break down this problem: {}\n"
        prob_prompt = ""
        sub_prob_prompt = ""
        sub_solu_prompt = ""
        idx = 0
        for sub_prob in processed_data['sub-problem-cot'].split('\n'):
            idx = idx + 1
            sub_prob = sub_prob.strip()
            sub_prob_prompt = sub_prob_prompt + sub_prob + ' '

        prob_prompt = test_problem_pattern.format(processed_data['problem'], sub_prob_prompt)
        prompt = prompt + prob_prompt
        
        return prompt, processed_data


    def process_single_data(self, data):
        processed_data = {
                'problem': data['question'],
                'cot': data['answer'].split('####')[0].strip(),
                'answer': data['answer'].split('####')[-1].strip(),
        }

        if ('sub-problem' in data):
            processed_data['sub-problem'] = data['sub-problem']
        if ('sub-problem-cot' in data):
            processed_data['sub-problem-cot'] = data['sub-problem-cot']
        if ('llm_sub_problem' in data):
            processed_data['sub-problem-cot'] = data['llm_sub_problem']

        if (self.is_clean == True):
            processed_data['cot'] = self.__clean(processed_data['cot'])
        return processed_data

    def __clean(self, content):
        pattern = '<<.*>>'
        results = re.findall(pattern, content)
        for span in results:
            content = content.replace(span, '')
        return content


class DataProcessForAQUA:
    def __init__(self, data_path, is_clean=False, num_examplar=5):
        self.num_examplar = num_examplar
        self.is_clean = is_clean
        self.data_path = data_path
        with open(self.data_path, 'r') as fin:
            raw_examplars = json.load(fin)

        self.examplars = []
        for examplar in raw_examplars:
            self.examplars.append(self.process_single_data(examplar))

    def process(
            self,
            data,
            num_del_examplar=0,
            pattern_demo="Problem: {}\nSolution: Let's think step by step. {} The answer is {}\n",
            pattern_test="Problem: {}\nSolution: Let's think step by step.",
    ):
        processed_data = self.process_single_data(data)

        prompt = ''
        num_example = self.num_examplar - num_del_examplar
        for i in range(num_example):
            problem = self.examplars[i]['problem']
            solution = self.examplars[i]['cot']
            answer = self.examplars[i]['answer']
            prompt = prompt + pattern_demo.format(problem, solution, answer)
        prompt = prompt + pattern_test.format(processed_data['problem'])

        return prompt, processed_data['answer']

    def process_single_data(self, data):
        processed_data = {
                'problem': data['question'] + 'Options: ' + ', '.join(data['options']),
                'cot': data['rationale'],
                'answer': data['correct']
        }
        return processed_data


class DataProcessForPENGUINS:
    def __init__(self, data_path, num_examplar=5):
        self.num_examplar = num_examplar
        self.data_path = data_path
        with open(self.data_path, 'r') as fin:
            raw_examplars = fin.readlines()
        self.examplars = ''.join(raw_examplars)
        self.examplars = self.examplars.strip() + '\n\n'

    def process(
            self, 
            data,
    ):
        prompt = self.examplars
        prompt = prompt + data['problem']

        return prompt, data['solution']


class DataProcessForCOLOR:
    def __init__(self, data_path, num_examplar=5):
        self.num_examplar = num_examplar
        self.data_path = data_path
        with open(self.data_path, 'r') as fin:
            raw_examplars = fin.readlines()
        self.examplars = ''.join(raw_examplars)
        self.examplars = self.examplars.strip() + '\n\n'

    def process(
            self, 
            data,
    ):
        prompt = self.examplars
        prompt = prompt + data['problem']

        return prompt, data['solution']


if __name__ == '__main__':
    data_processer = DataProcessForMATH('demo/math.json', is_clean=True)
