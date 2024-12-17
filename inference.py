import time
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

system_prompt = '''
营商环境政策汇编（2023）- 上海:
（五）获取金融服务。推动完善担保交易、电子支付等领域政策法规和绿色融资领域相
五、继续夯实工作基础和监督保障（十四）落实政府信息公开审查制度。健全完善公开属性认定流程，在保障公民、法人和其他组织依法获取政府信息的同时，坚持“先审查、后公开”和“一事一审”原则，严防泄密和相关风险，加强对个人隐私、商业秘密的保护。把握好政策发布的时度效，合理引导社会和市场预期。
（八）提升涉案信息线下查询渠道便利度。各相关部门和单位支持管理人持人民法院受理破产申请裁定书、指定破产管理人决定书在各区自然资源确权登记事务中心、市公积金中心各管理部、市公安局交通警察总队车管所、税务机关办税服务厅注销清税业务专窗（专区）等线下窗口查询破产企业相关信息。鼓励各区及市有关部门通过整合线下查询窗口，开辟管理人查询“绿色通道”等方式进一步便利管理人通过线下方式查询涉案信息。（责任单位：市高院、市公安局、市规划资源局、上海银保监局、上海证监局、市税务局、市人力资源社会保障局、市住房城乡建设管理委、市通信管理局等；各区政府）
2.完善综合竣工验收制度。结合不同类型项目特点，完善建筑工程综合竣工验收制度，进一步明确综合竣工验收中各专业验收管理部门附条件通过的办理要求，精简提前查看的实施范围，规范提前查看服务流程，编制《上海市建筑工程综合竣工验收工作手册（2023版）》，推进综合竣工验收标准化、规范化、便利化。细化建筑工程配建公交和停车设施等专业验收要求，压缩出入口专项验收办理时限，精简优化办事流程。在交通枢纽、水务、绿化工程中探索推行综合竣工验收。（责任部门：市住房城乡建设管理委、市规划资源局、市水务局、市绿化市容局、市卫生健康委、市国动办、市交通委（市道运局）、市交警总队、市气象局）
第七条【信息公示】施工许可证应当放置在施工现场备查。除保密工程外，建设单位应当在施工现场的施工铭牌向社会公示施工许可证信息。公示信息包括但不限于：建设单位、工程名称、建设地址、建设规模、合同工期、参建单位及其项目负责人、施工许可证编号、发证机关、单体（位）工程明细、二维验证码图片等。
区政府决策部署，认真贯彻《2023年上海公安政务公开工作要点》，以习近平新时代中国特色社会主义思想为指导，深入贯彻落实党的二十大精神，认真落实国务院办公厅和上海市关于深化新时代政务公开的工作要求，逐步转变政务公开职能，统筹政务公开和安全保密，对标上级工作部署要求，努力补齐工作短板，有效提升公信力和执行力，增强公开信息的实用性，显现政民互动效果，助力上海公安工作高质量发展。
优化智能预填、智能预审、智能审批等功能，推进政务服务“好办”“快办”。构建线上线下全面融合的“泛在可及”服务体系，推动PC端、大厅端、移动端、自助端“四端”联动及功能升级。创新虚拟政务服务窗口办理，提供与实体大厅“同标准、同服务”的办理体验，将实体大厅窗口向园区延伸。围绕企业和个人全生命周期，积极拓展新增“一件事一次办”主题服务。依托企业专属网页、“随申办”企业云平台，加大惠企政策归集力度。优化“一网通办”惠企政策专区，集中发布涉企政策要点。推动行政给付、资金补贴扶持、税收优惠等政策实现“免申即享”。探索在行政服务中心大厅设立惠企政策综合窗口，统一受理政策
（99）深化示范区行政执法协同，创新案件信息互通、案件成果互享、人员交流互动等
国家及各省市营商环境政策汇编（2023）报投诉处理机制。统筹协调公平竞争审查工作，完善并推进公平竞争审查联席会议工作任务。做好竞争合规宣传和倡导。

以上是你可以参考的信息，可能与后面的用户问题相关也有可能不相关，请你自行判断。
如果有关，请严格根据上述提供内容进行回答，严禁使用任何你自己的知识。
如果无关，请你拒绝回答用户的问题，不要回答。
'''

user_prompt = '''
请告诉我上海的营商政策有哪些？
'''

start_time = time.time()
print('Input length:', len(system_prompt) + len(user_prompt))
print()
print('Start at:', start_time)

model_path = "/data/disk1/guohaoran/model/self/Qwen2.5-3B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float32,
    device_map="cuda:7",
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

loaded_time = time.time()
print()
print('Loaded model at:', loaded_time)
print('Load duration:', loaded_time - start_time)

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print()
finished_time = time.time()
print('Finished at:', finished_time)
print('Inference duration:', finished_time - loaded_time)
print('\n')
print(response)
print()
print('Response length:', len(response))
