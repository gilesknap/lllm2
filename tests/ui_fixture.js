// Representative API fixtures for test_ui_browser.cjs.
window.fixture={saved:null,posts:[],delay:0,disconnected:false,checkError:'',downloads:[],results:[]};
const sample={model:'/models/Qwen3-8B/model.gguf',engine:'/engines/llama-server',backend:'CUDA',device:'CUDA0',context:32768,slots:1,gpu_layers:null,flash:'on',cache:'q8_0',cache_k:null,cache_v:null,speculation:'none',drafter:'',pair_confirmed:false,draft_length:3,effort:'default',draft_cache:'q8_0',chat_template:'',batch_size:null,ubatch_size:null,backend_sampling:false,cuda_graph_opt:'default',cache_ram_mib:null,context_checkpoints:null,lookup_ngram_n:null,lookup_ngram_m:null};
fixture.settings=sample;
fixture.catalog=[{id:'qwen3-8b',name:'Qwen3-8B',repo:'Qwen/Qwen3-8B-GGUF',file:'model.gguf',size_gb:5,max_ctx:131072},{id:'dense',name:'Qwen3.8-27B',repo:'unsloth/Qwen3.8-27B-GGUF',file:'dense.gguf',size_gb:15.36,recommendation:{rank:1,label:'Recommended',description:'Our default starting model.',variant:'UD-Q4_K_S'}},{id:'moe',name:'Qwen3.6-35B-A3B',repo:'unsloth/Qwen3.6-MTP-GGUF',file:'moe.gguf',size_gb:18.21,recommendation:{rank:2,label:'Faster alternative',description:'Faster generation in our tested RTX 3090 coding workloads.',variant:'MTP build · IQ4_XS'}}];
fixture.models=[{path:sample.model,catalog_id:'qwen3-8b',identity_verified:false,metadata:{context:131072}}];
fixture.engines=[{path:sample.engine,devices:['CUDA0']}];
fixture.engine={running:false,ready:false};
window.fetch=async (path,options={})=>{const data=options.body?JSON.parse(options.body):null;if(data)fixture.posts.push({path,data});if(fixture.disconnected)throw Error('Offline');if(fixture.delay&&path!='/api/status')await new Promise(r=>setTimeout(r,fixture.delay));let body={};switch(path){
case '/api/status':body={token:'test',version:'1.2.3-test',hardware:{gpus:[{name:'RTX A1000',used_mib:2048,total_mib:8192}],ram_gib:62,ram:{total_gib:62,used_gib:18,available_gib:44}},engine:fixture.engine,job:fixture.job||{status:'idle'},downloads:fixture.downloads,endpoint:'http://127.0.0.1:1920/v1',paths:{models:'/models',engines:['/engines']}};break;
case '/api/catalogue':body={entries:fixture.catalog};break;
case '/api/models/find':body={entries:fixture.findEntries||[],repositories:2,fetched_at:Date.now()/1000};break;
case '/api/catalogue/add':fixture.catalog.push(fixture.findEntries.find(e=>e.id===data.id));break;
case '/api/catalogue/removal-preview':body={name:'Test model',paths:['/models/test/model.gguf']};break;
case '/api/catalogue/remove':fixture.catalog=fixture.catalog.filter(e=>e.id!==data.id);break;
case '/api/discover':body={models:fixture.models,catalog:fixture.catalog,engines:fixture.engines};break;
case '/api/launch/select':body=fixture.models.length?{settings:{...sample,model:data.model||sample.model},source:'Estimated starting settings',notes:['Automatic fitting can use system RAM.']}:{reason:'Choose or download a model.'};break;
case '/api/capabilities':body={features:{},engine:{devices:['CUDA0']}};break;
case '/api/launch/check':body={valid:!fixture.checkError,error:fixture.checkError,saved_exists:!!fixture.saved};break;
case '/api/default/resolve':body={settings:data.source==='saved'?fixture.saved:sample,source:data.source==='saved'?'Saved defaults · manual preferences':'Estimated starting settings'};break;
case '/api/default/save':fixture.saved=data.result_id?fixture.results.find(r=>r.id===data.result_id).settings:data.settings;body=fixture.saved;break;
case '/api/results/delete':{const deleted=fixture.results.filter(r=>data.result_ids.includes(r.id)).map(r=>r.id);fixture.results=fixture.results.filter(r=>!deleted.includes(r.id));body={deleted};break;}
case '/api/results':body=fixture.results;break;
case '/api/results/export':body=fixture.fullResults||fixture.results;break;
case '/api/result/preview':{const r=fixture.results.find(r=>r.id===data.result_id);body={settings:{...r.settings,...(data.use_context?{context:(data.reserve_headroom?r.recommended_context:r.largest_observed_context)*r.settings.slots}:{})},source:'Experiment result',result_id:r.id,use_context:!!data.use_context,reserve_headroom:!!data.reserve_headroom,evidence:{kind:'benchmark'},notes:[]};break;}
case '/api/start':fixture.engine={running:true,ready:true,settings:data.settings,pid:123};break;
}return {ok:true,json:async()=>structuredClone(body)};};
