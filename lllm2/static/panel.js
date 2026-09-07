const $=id=>document.getElementById(id), esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const settingKeys=['model','engine','backend','device','context','slots','gpu_layers','flash','cache','cache_k','cache_v','speculation','drafter','pair_confirmed','draft_length','effort','draft_cache','chat_template','batch_size','ubatch_size','backend_sampling','cuda_graph_opt','cache_ram_mib','context_checkpoints','lookup_ngram_n','lookup_ngram_m'];
const optionalNumbers=['batch_size','ubatch_size','cache_ram_mib','context_checkpoints','lookup_ngram_n','lookup_ngram_m'];
const optionalCache=['cache_k','cache_v'];
const cachePair=s=>[s.cache_k||s.cache,s.cache_v||s.cache];
const cacheLabel=s=>{const [k,v]=cachePair(s);return `K ${k} / V ${v}`;};
const settingValue=(s,k)=>optionalNumbers.includes(k)||optionalCache.includes(k)?s[k]??null:k==='backend_sampling'?s[k]??false:k==='cuda_graph_opt'?s[k]??'default':s[k];
// Shared launch and feature copy; help buttons stay outside labels.
const launchHelp={
 model:['Installed checkpoint','The GGUF file contains the model weights. Size and precision affect memory, speed and answers. Keep the recommended checkpoint unless comparing models.'],
 engine:['llama-server binary','The program that runs the checkpoint. Builds support different devices and features; changing it can affect speed and compatibility. Keep a working build for fair comparisons.'],
 backend:['Backend','CUDA and Vulkan are ways to run the model on your NVIDIA GPU. Support and speed can differ; keep the recommended backend unless comparing them.'],
 device:['GPU device','The GPU reported by this engine for the chosen backend. Its free memory limits what fits. Keep the detected device unless deliberately using another GPU.'],
 context:['Total allocated context','Text capacity in tokens, including prompts and replies, shared across slots. More needs more memory and can take longer to fill. Keep the recommended allocation unless you need a different context.'],
 slots:['Slots','How many requests can hold their own conversation state. Total context is divided between slots; more slots leave less per request. This is not CPU threads. Keep the recommended value for normal use.'],
 batch_size:['Logical batch (tokens)','Maximum prompt tokens submitted together. A larger batch may read prompts faster but uses more working memory. Leave blank for this engine’s default; microbatch must not exceed it.'],
 backend_sampling:['Target GPU sampling','Experimental: choose the next target token on the GPU. This may reduce overhead, but unsupported sampler requests can fall back. Leave unchecked for the engine default until measured. Draft sampling is unchanged.'],
 cuda_graph_opt:['Concurrent CUDA streams','Experimental: overlap eligible operations inside a CUDA graph. Requires one visible CUDA device and ordinary CUDA Graphs. Default preserves the engine environment; this does not add concurrent requests.'],
 cache_ram_mib:['Host prompt cache (MiB)','Finite RAM allowance for saved conversations. Zero disables saved conversation caching. Blank keeps the normal engine default; the warm experiment uses 2048 MiB when blank. This is not a total process RAM cap.'],
 context_checkpoints:['Context checkpoints per slot','Saved states can help resume edited history on recurrent models. Zero disables these checkpoints. Blank keeps the normal engine default; the warm experiment uses four when blank. More checkpoints use additional memory.'],
 ubatch_size:['Physical microbatch (tokens)','Prompt tokens processed together in one physical chunk of the logical batch. Larger chunks may improve prefill speed but need more GPU memory. Leave blank for the engine default until compared.'],
 gpu_layers:['GPU layers','How much of the model runs on the GPU. 999 requests all layers; fewer can save GPU memory but may slow inference. Keep the starting value if the model fits.'],
 flash:['Flash attention','Calculates attention with less working memory and may improve speed. Auto leaves the choice to the engine. Keep the recommended value; quantized KV cache and DFlash require on here.'],
 cache:['Common conversation memory precision','Explicitly selecting this links K and V and clears Advanced overrides. The actual pair is shown below. Stores information about text already read. q8_0 and q4_0 use less memory than f16; speed and accuracy can change. Keep the starting precision until compared.'],
 effort:['Reasoning effort','Asks a compatible model/template to spend more or less effort reasoning. More can take longer; accepted values depend on the template. Keep default unless testing a supported value.'],
 lookup_ngram_n:['Lookup match N','Number of matching tokens used by ngram-simple. Set N and M together with 1 ≤ N ≤ M; N greater than M produces no drafts on the verified build. Blank preserves old modes and selects N3 for MTP+lookup.'],
 lookup_ngram_m:['Lookup draft M','Lookup continuation width, independent of MTP draft length. Set together with N. Blank preserves old modes and selects M3 for MTP+lookup. Compare equal widths first; extra guesses can add rejected work.'],
 'draft-mtp,ngram-simple':['MTP + lookup · experimental','Explicit combination requiring both engine methods, lookup controls and checkpoint MTP tensors. Build 662a0b0 tries lookup before MTP fallback. Ordinary HTTP counters aggregate both methods; no speedup is implied.'],
 speculation:['Guess ahead (speculative decoding)','Checks guessed tokens together to try to generate answers faster. Choose MTP, DFlash, prompt lookup or the explicit MTP+lookup experiment; overhead can outweigh savings. Keep the starting mode until measured.'],
 'draft-mtp':['MTP','Uses a prediction head inside the checkpoint to guess upcoming tokens. It needs MTP tensors and engine support. Longer guesses can add wasted work; keep the starting draft length.'],
 'draft-dflash':['DFlash · experimental','Uses a separate target-specific model to guess tokens. Extra memory and checking work can outweigh savings. Requires CUDA, flash attention on and a verified pair; use only for a deliberate comparison.'],
 'ngram-simple':['Prompt lookup (ngram)','Guesses continuations from repeated text, without a draft model. It may help copying code but adds checking work when guesses miss. Leave the starting mode unless testing repetitive tasks.'],
 draft_length:['Draft length','Maximum tokens guessed at once when the engine supports this control. Longer guesses may help, or waste more work when rejected. Keep the starting length; unused with speculation off.'],
 cache_k:['K cache override','Inherit uses common precision; explicit values override only attention K. Mixed pairs require engine/backend verification. Draft cache is unchanged.'],
 cache_v:['V cache override','Inherit uses common precision; explicit values override only attention V. Quantized cache requires flash attention on here. Mixed types may lack a GPU kernel.'],
 cache_pair:['Actual cache pair','Resolved K and V include Advanced overrides. Compiled kernel evidence is separate from runtime execution and measured capacity.'],
 draft_cache:['Draft KV cache','Conversation-memory precision for MTP or DFlash guesses. Lower precision may save memory but needs engine support and can affect speed. Default leaves it to the engine; keep the starting value.'],
 chat_template:['Chat template override (optional)','A local template file formats messages and reasoning instructions. A mismatch can cause errors or change answers. Keep the supplied template; blank uses the checkpoint template.'],
 drafter:['DFlash drafter GGUF','Local path to the extra model used only by DFlash. It must match this exact target and uses extra memory. Browse selects a file; it does not prove compatibility. Leave alone unless testing DFlash.'],
 pair_confirmed:['Verify the DFlash pair','Confirm training and conversion compatibility for this exact target. A matching filename is not evidence; a wrong pair can fail or behave incorrectly. Leave unchecked unless verified.']
};
function helpTip(key,prefix){const [name,description]=launchHelp[key]||[key,'Check engine and model compatibility before comparing performance.'];const id=`${prefix}-tip-${key}`;return `<span class="tip"><button type="button" aria-label="${esc(name)} help" aria-describedby="${id}" aria-expanded="false">ⓘ</button><span class="tip-content" role="tooltip" id="${id}">${esc(description)}</span></span>`;}
function launchTips(){
 for(const key of settingKeys){
  const control=$(key), label=control.closest('label')||document.querySelector(`label[for="${key}"]`);
  if(control.type==='checkbox'){const row=document.createElement('div');row.className='row wide';label.replaceWith(row);row.append(label);row.insertAdjacentHTML('beforeend',helpTip(key,'launch'));continue;}
  label.htmlFor=key;
  if(label.contains(control)){const field=document.createElement('div');field.className='field-control'+(label.classList.contains('wide')?' wide':'');label.replaceWith(field);const title=document.createElement('div');title.className='field-title';field.append(title);title.append(label);field.append(control);for(const child of [...label.children])field.append(child);label.textContent=launchHelp[key][0];title.insertAdjacentHTML('beforeend',helpTip(key,'launch'));}
  else{label.textContent=launchHelp[key][0];label.insertAdjacentHTML('afterend',helpTip(key,'launch'));}
 }
}
let featureState=null;
function renderFeatures(){
 if(!featureState)return;
 const s=settings(), statuses={experimental:'Experimental · runtime unverified',available:'Available to try',unsupported:'Not supported',unknown:'Support unconfirmed','missing prerequisites':'Needs setup'};
 $('features').innerHTML=Object.entries(featureState).map(([key,v])=>{
  const choice=key in s?`Form value: ${s[key]??'engine default'}`:s.speculation===key?'Selected in form':'Not selected in form';
  const reason=key==='flash'||key==='cache'||key==='effort'?(v.status==='unsupported'?'This engine does not advertise the required control.':v.status==='unknown'?'Support could not be confirmed for this engine and template.':'Engine control detected; this model/backend still needs a successful launch.'):v.reason;
  return `<div class="feature"><div class="field-title"><b>${esc((launchHelp[key]||[key])[0])}</b>${helpTip(key,'feature')}</div><div>${esc(statuses[v.status]||'Support unconfirmed')} · ${esc(choice)}</div><small>${esc(reason)}</small><details><summary>Technical diagnostics</summary><small>${esc(key)} · ${esc(v.status)}</small><small>${esc(v.reason)}</small></details></div>`;
 }).join('');
 initTips($('features'));
}
let token='', discovered={}, combinations=[], results=[], lastResults='', statusState={}, loadedDefaults=null;
let view='launch', drafts={launch:null,experiments:null}, connected=false, resolving=true, selectionSequence=0, validationSequence=0;
let validationSnapshot='', validationError='', actionError='', selectionNote='', pendingAction=false, pollPending=false, scanPending=false;
let savedExists=false, engineOverride=false, editTimer=null, lastDownloads='', startAttempt=null, discoveredOnce=false, queuedView=null;
const sameSettings=(a,b)=>!!a&&!!b&&settingKeys.every(k=>settingValue(a,k)===settingValue(b,k));
const snapshot=()=>JSON.stringify(settings());
const currentLaunch=()=>view==='launch'?settings():drafts.launch?.settings;
const modelName=path=>{const m=discovered.models?.find(m=>m.path===path),c=discovered.catalog?.find(c=>c.id===m?.catalog_id);return c?(c.recommendation?.name||c.name):path?.split('/').slice(-2).join('/')||'No model selected';};

function settings(){return Object.fromEntries(settingKeys.map(k=>[k,$(k).type==='checkbox'?$(k).checked:$(k).type==='number'?(optionalNumbers.includes(k)&&$(k).value===''?null:Number($(k).value)):optionalCache.includes(k)?$(k).value||null:$(k).value]));}
function slotNote(){$('slot-note').textContent=`${Math.floor(Number($('context').value)/Number($('slots').value)).toLocaleString()} tokens per conversation (slot), including the reply. Total allocation is divided across ${$('slots').value} slot(s).`;benchmarkCost();}
function message(s,error=false){$('message').textContent=s;$('message').style.display='block';$('message').className=error?'error':'';}
async function api(path,data){const r=await fetch(path,{signal:AbortSignal.timeout(path==='/api/start'?15000:path==='/api/status'?10000:180000),...(data===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json','X-LLLM2-Token':token},body:JSON.stringify(data)})});const d=await r.json();if(!r.ok)throw Error(d.error||r.statusText);return d;}
async function attempt(fn){try{await fn();}catch(e){message(e.message,true);}}
async function refreshResults(){
 results=await api('/api/results');const key=JSON.stringify(results.map(r=>[r.id,r.status,r.samples.length,r.probes.length,r.finished]));if(key===lastResults)return;lastResults=key;
 const fmt=x=>x==null?'—':Number(x).toLocaleString(undefined,{maximumFractionDigits:1});
 $('results').innerHTML=results.flatMap(r=>(r.samples.length?r.samples:[null]).map((s,i)=>{
  const warm=r.measurement_mode==='warm-conversation';
  return `<tr><td><b>${esc(r.label)}</b> · ${esc(warm?(s?.control?'Uncached replay':'Conversation')+' · '+(s?.turn||r.status):s?.workload||r.status)}<small>${warm?'Warm experiment':'Cold benchmark'} · ${esc(s?.status||r.status)}</small><small>${esc(r.settings.model.split('/').slice(-2).join('/'))}</small><small>${esc(r.started)} · ${esc(r.settings.backend)} · ${esc(r.status)}</small><small>${esc(s?.error||r.error||'')}</small>${s?.adherence?`<small>Source adherence: ${esc(s.adherence.status)}</small>`:''}</td>
  <td>${fmt(warm?s?.processed_prefill_tok_s:s?.prefill_tok_s)}${warm?'<small>Processed tokens only</small>':''}</td>
  <td>${fmt(s?.decode_tok_s)}${warm?`<small>First token event: ${fmt(s?.first_token_event_seconds)} s<br>First text: ${fmt(s?.first_text_seconds)} s<br>Completion: ${fmt(s?.completion_seconds)} s</small>`:''}</td>
  <td>${fmt(s?.input_tokens)} / ${fmt(s?.output_tokens)}${s?.adherence?`<small>Output requested / cap: ${fmt(s.requested_output_budget)} / ${fmt(s.output_budget)} · total ${fmt(s.wall_seconds)} s</small>`:''}${warm?`<small>Processed ${fmt(s?.processed_tokens)} · reused ${fmt(s?.reused_tokens)}</small>`:''}</td>
  <td>${fmt(s?.peak_total_gpu_used_mib)}${s?.peak_engine_rss_mib!=null?`<small>Engine RSS: ${fmt(s.peak_engine_rss_mib)} MiB sampled peak</small>`:''}</td>
  <td>${fmt(r.largest_observed_context)} / ${fmt(r.recommended_context)}<small>${esc(r.context_search_stop_reason||'')}${r.context_search_status==='complete'&&r.context_failed_upper_bound?`Search range: ${fmt(r.largest_observed_context)}–${fmt(r.context_failed_upper_bound)} tokens. `:''}${r.context_ceiling_reached?'Search ceiling reached; maximum may be higher.':''}</small></td>
  <td><details><summary>${esc(r.settings.speculation)} · ${esc(cacheLabel(r.settings))}</summary><pre>${esc(JSON.stringify({settings:r.settings,cache_settings:s?.cache_settings||r.cache_settings,batch_settings:s?.batch_settings||r.batch_settings,execution_settings:s?.execution_settings||r.execution_settings,engine:r.engine,options:r.options,timings:s?.timings,speculative_settings:s?.speculative_settings,adherence:s?.adherence,host_before:s?.host_before,host_after:s?.host_after,context_search_resolution:r.context_search_resolution,context_failed_upper_bound:r.context_failed_upper_bound,context_seed_from_speed_sample:r.context_seed_from_speed_sample,probes:r.probes.map(p=>({context:p.context_per_slot,status:p.status,error:p.error}))},null,2))}</pre></details>${!warm&&i===0&&r.status==='complete'&&r.quality_status!=='failed'?`<button data-promote="${esc(r.id)}">Try in Launch</button>${r.recommended_context?`<button data-context="${esc(r.id)}">Try with headroom context</button>`:''}`:''}</td></tr>`;
 })).join('')||'<tr><td colspan="7">No experiments yet.</td></tr>';
}
function resetExperimentCeiling(modelPath=$('model').value){
 const model=discovered.models?.find(m=>m.path===modelPath);
 const entry=discovered.catalog?.find(m=>m.file===modelPath.split('/').pop());
 const ceiling=Math.min(model?.metadata.context||entry?.max_ctx||131072,1048576);
 $('max_context').value=ceiling;
 $('search-limit-note').textContent=`Search ceiling starts at ${ceiling.toLocaleString()} tokens per slot (${model?.metadata.context?'GGUF metadata':entry?.max_ctx?'catalogue limit':'generic fallback; model limit unknown'}). This is an upper bound to test, not a measured usable context. You can lower it.`;
}
function benchmarkCost(){
 const quick=$('sweep_prompts').checked, full=$('full_window').checked, search=$('search_context').checked;
 $('prompt_tokens').disabled=quick;
 $('max_context').disabled=$('context_timeout').disabled=!search;
 const hasSource=[...document.querySelectorAll('#workloads input:checked')].some(x=>['source-copy','source-small-edit','context-retrieval-edit'].includes(x.value));
 const upper=Math.floor(Number($('context').value)/Number($('slots').value))-Math.max(Number($('output_tokens').value),hasSource?2048:0)-32;
 const sizes=[...new Set([...(quick?[1024,16384,65536].map(n=>Math.min(n,upper)):[Number($('prompt_tokens').value)]),...(full?[upper]:[])])].sort((a,b)=>a-b);
 const workloads=document.querySelectorAll('#workloads input:checked').length, repeats=Number($('repeats').value);
 if(!Number.isFinite(upper)||upper<128){$('benchmark-time-tip').textContent='Adjust the prompt and output budgets to estimate timing.';$('benchmark-cost').textContent='Increase launch context or reduce output tokens to fit a real prompt.';return;}
 $('benchmark-cost').textContent=`Per configuration: ${sizes.length*workloads*repeats} speed samples (${sizes.map(n=>n.toLocaleString()).join(' / ')} input tokens), plus ${search?'up to 8 context probes':'no context search'}.`;$('benchmark-time-tip').textContent=(quick&&!full&&!hasSource&&Number($('output_tokens').value)===256&&workloads>0?`Rough speed-test estimate: ${2*workloads*repeats}–${3*workloads*repeats} minutes; smaller windows may finish sooner. `:'')+(full?'Full-window time is included in the sample count; see the full-window option’s timing tooltip. ':'')+'Suites multiply this work across configurations. Estimates are based on the observed Qwen/Vulkan run, not a time limit.';
}
function resetExperiments(){
 for(const control of $('experiments').querySelectorAll('input')){
  if(control.type==='checkbox')control.checked=control.defaultChecked;
  else control.value=control.defaultValue;
 }
 resetExperimentCeiling();
 benchmarkCost();
 message('Experiment settings reset to defaults for the selected model.');
}
$('reset-experiments').onclick=resetExperiments;
for(const k of ['sweep_prompts','full_window','search_context','prompt_tokens','output_tokens','repeats'])$(k).addEventListener('input',benchmarkCost);
document.querySelectorAll('#workloads input').forEach(el=>el.addEventListener('input',benchmarkCost));
benchmarkCost();
function positionTip(tip){
 const content=tip.querySelector('.tip-content'), rect=tip.querySelector('button').getBoundingClientRect();
 const width=Math.min(280,window.innerWidth-24);
 content.style.left=Math.max(12,Math.min(rect.left,window.innerWidth-width-12))+'px';
 content.style.top=Math.max(12,Math.min(rect.bottom+6,window.innerHeight-content.offsetHeight-12))+'px';
}
function initTips(root=document){root.querySelectorAll('.tip').forEach(tip=>{
 if(tip.dataset.initialized)return;tip.dataset.initialized='true';
 const button=tip.querySelector('button');
 tip.addEventListener('mouseenter',()=>{tip.classList.remove('dismissed');positionTip(tip);});
 tip.addEventListener('mouseleave',()=>tip.classList.remove('dismissed'));
 button.addEventListener('focus',()=>{tip.classList.remove('dismissed');positionTip(tip);});
 button.addEventListener('click',e=>{e.preventDefault();const open=tip.classList.toggle('open');button.setAttribute('aria-expanded',String(open));tip.classList.toggle('dismissed',!open);if(!open)button.blur();else positionTip(tip);});
 button.addEventListener('keydown',e=>{if(e.key==='Escape'){tip.classList.remove('open');tip.classList.add('dismissed');button.setAttribute('aria-expanded','false');button.blur();}});
});}
launchTips();initTips();
for(const event of ['resize','scroll'])window.addEventListener(event,()=>document.querySelectorAll('.tip').forEach(tip=>{if(tip.querySelector('.tip-content').getClientRects().length)positionTip(tip);}),true);
document.addEventListener('click',e=>{document.querySelectorAll('.tip.open').forEach(tip=>{if(!tip.contains(e.target)){tip.classList.remove('open');tip.querySelector('button').setAttribute('aria-expanded','false');tip.querySelector('button').blur();}});});
let fileListing=null, selectedFile='', fileRequest=0;
function renderFiles(){
 const filter=$('file-filter').value.toLowerCase();
 const entries=(fileListing?.entries||[]).filter(e=>e.name.toLowerCase().includes(filter));
 $('file-list').replaceChildren();
 for(const entry of entries){
  const button=document.createElement('button');
  button.type='button';button.setAttribute('aria-pressed',String(selectedFile===entry.path));
  const name=document.createElement('span');name.textContent=(entry.directory?'📁 ':'')+entry.name;
  const detail=document.createElement('small');detail.textContent=entry.directory?'Folder':(entry.size/2**30).toFixed(2)+' GiB';
  button.append(name,detail);
  button.onclick=()=>{if(entry.directory){openFolder(entry.path);}else{selectedFile=entry.path;$('file-selected').textContent=entry.path;$('file-use').disabled=false;renderFiles();}};
  button.ondblclick=()=>{if(!entry.directory)useFile(entry.path);};
  $('file-list').append(button);
 }
 if(!entries.length)$('file-list').textContent='No matching folders or GGUF files.';
}
async function openFolder(directory){
 const request=++fileRequest;selectedFile='';fileListing=null;
 $('file-use').disabled=true;$('file-selected').textContent='No file selected.';
 $('file-list').textContent='Loading…';$('file-error').textContent='';
 try{
  const listing=await api('/api/files',{directory});
  if(request!==fileRequest||!$('file-dialog').open)return;
  fileListing=listing;$('file-directory').value=listing.directory;$('file-filter').value='';
  $('file-up').disabled=listing.parent===listing.directory;renderFiles();
 }catch(e){if(request===fileRequest){$('file-error').textContent=e.message;$('file-list').replaceChildren();}}
}
function closeFiles(){fileRequest++;$('file-dialog').close();}
function useFile(path){
 $('drafter').value=path;$('pair_confirmed').checked=false;
 closeFiles();edited('drafter');
}
$('browse-drafter').onclick=()=>{
 $('file-dialog').showModal();
 const current=$('drafter').value.trim(), slash=current.lastIndexOf('/');
 openFolder(slash>=0?(current.slice(0,slash)||'/'):null);
};
$('file-up').onclick=()=>{if(fileListing)openFolder(fileListing.parent);};
$('file-home').onclick=()=>openFolder(fileListing?.home||'~');
$('file-models').onclick=()=>openFolder(null);
$('file-go').onclick=()=>openFolder($('file-directory').value);
$('file-directory').onkeydown=e=>{if(e.key==='Enter'){e.preventDefault();openFolder(e.target.value);}};
$('file-filter').oninput=renderFiles;
$('file-use').onclick=()=>{if(selectedFile)useFile(selectedFile);};
for(const id of ['file-close','file-cancel'])$(id).onclick=closeFiles;
$('file-dialog').addEventListener('cancel',()=>{fileRequest++;});
$('export').onclick=()=>attempt(async()=>{const full=await api('/api/results/export');const url=URL.createObjectURL(new Blob([JSON.stringify(full,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='lllm2-results.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});


// One editor, moved between views, with independent in-memory drafts and provenance.
function fill(s){
 if(s.model&&!Array.from($('model').options).some(o=>o.value===s.model))$('model').add(new Option(modelName(s.model),s.model));
 for(const k of settingKeys){
  if(k==='device')continue;
  if($(k).type==='checkbox')$(k).checked=s[k]??false;
  else $(k).value=s[k]??(k==='cuda_graph_opt'?'default':'');
 }
 $('device').replaceChildren(new Option(s.device||'Choose device',s.device||''));
 slotNote();defaultState();launchState();
}
function rememberDraft(){drafts[view]={settings:settings(),defaults:loadedDefaults,engineOverride,savedExists,selectionNote};}
async function switchView(next){
 if(next===view)return;
 if(resolving||scanPending){queuedView=next;return;}
 rememberDraft();selectionSequence++;validationSequence++;clearTimeout(editTimer);resolving=false;validationSnapshot='';
 view=next;
 if(!drafts[next])drafts[next]=structuredClone(drafts.launch);
 const d=drafts[next];loadedDefaults=d.defaults;engineOverride=d.engineOverride;savedExists=d.savedExists;selectionNote=d.selectionNote||'';
 $('launch-view').hidden=next!=='launch';$('experiments-view').hidden=next!=='experiments';
 $('nav-launch').setAttribute('aria-current',next==='launch'?'page':'false');$('nav-experiments').setAttribute('aria-current',next==='experiments'?'page':'false');
 $(next==='launch'?'editor-host-launch':'experiment-settings-host').append($('settings-editor'));
 $(next==='launch'?'summary-host-launch':'summary-host-experiments').append($('source-summary'));
 if(next==='launch')$('selection-note').before($('all-models'));else $('summary-host-experiments').before($('all-models'));
 $('customize').querySelector('summary').textContent=next==='launch'?'Customize settings':'Customize experiment settings';
 $('save-default').hidden=next==='experiments';
 fill(d.settings);renderRecommendations();await inspect();
 if(next==='experiments')await refreshResults();
}
for(const [id,next] of [['nav-launch','launch'],['nav-experiments','experiments']])$(id).onclick=e=>{e.preventDefault();location.hash=next;attempt(()=>switchView(next));};
window.addEventListener('hashchange',()=>attempt(()=>switchView(location.hash==='#experiments'?'experiments':'launch')));
function finishNavigation(){if(queuedView&&!resolving&&!scanPending){const next=queuedView;queuedView=null;attempt(()=>switchView(next));}}
function defaultState(){
 if(!$('summary-title'))return;
 const s=settings(),changed=loadedDefaults?settingKeys.filter(k=>settingValue(s,k)!==settingValue(loadedDefaults.settings,k)):[];
 const mode=changed.length?'custom':loadedDefaults?.mode||'custom';
 const names={recommended:'Recommended settings',saved:'My saved settings',custom:'Custom settings',running:'Current model settings',result:'Experiment settings · for next launch'};
 const source=loadedDefaults?.source||'Choose a model to resolve its starting settings.';
 const measured=source.startsWith('Measured built-in');
 const chosen=discovered.models?.find(m=>m.path===s.model);
 $('selected-model').hidden=!s.model||view!=='launch'||!!(chosen?.identity_verified&&discovered.catalog?.find(m=>m.id===chosen.catalog_id)?.recommendation);
 $('selected-model').textContent='Selected model: '+modelName(s.model);
 $('summary-title').textContent=resolving?'Preparing settings…':names[mode];
 $('summary-context').textContent=s.model?`${Math.floor(s.context/s.slots).toLocaleString()} tokens per conversation · ${s.slots} slot${s.slots===1?'':'s'} · ${s.backend}${mode==='recommended'?' · '+(measured?(source.includes('qualified')?'Qualified recommendation':'Tested on RTX 3090'):'Estimated starting settings'):''}`:'';
 $('draft-note').textContent=changed.length?'Changed: '+changed.map(k=>launchHelp[k][0]).join(', ')+'. Last loaded evidence does not validate these edits.':(s.speculation.includes('dflash')||s.speculation.includes(',')||cachePair(s)[0]!==cachePair(s)[1])?'Experimental settings selected. Review feature details before starting.':'';
 $('default-source').textContent=source;
 $('recommendation-notes').replaceChildren(...(loadedDefaults?.notes||[]).map(note=>{const p=document.createElement('p');p.textContent=note;return p;}));
 $('default-evidence').hidden=!loadedDefaults?.evidence;
 $('default-evidence').querySelector('summary').textContent=changed.length?'Last loaded evidence':'Recommendation evidence';
 $('default-evidence').querySelector('pre').textContent=loadedDefaults?.evidence?JSON.stringify(loadedDefaults.evidence,null,2):'';
 $('load-default').hidden=!savedExists;$('load-default').disabled=resolving;
 $('built-in-default').hidden=mode==='recommended';$('built-in-default').disabled=resolving||!s.model;
 $('saved-note').textContent=savedExists&&mode==='recommended'?'Recommended settings are selected. Your saved settings are still available.':'';
 $('experiment-model').textContent=modelName(s.model);
 for(const id of ['drafter','pair_confirmed']){
  const el=id==='pair_confirmed'?$(id).closest('label').parentElement:$(id).closest('.wide');if(el)el.hidden=s.speculation!=='draft-dflash';
 }
}
function launchState(){
 const s=settings(), launch=currentLaunch(),running=statusState.engine||{},job=statusState.job||{};
 const same=sameSettings(launch,running.settings),launchJob=job.active&&job.kind==='launch',experiment=job.active&&job.kind!=='launch';
 $('cache-summary').textContent=`Actual attention cache: ${cacheLabel(s)}. ${cachePair(s)[0]!==cachePair(s)[1]?'Mixed precision is experimental; measured capacity and quality are not established.':''}`;
 $('batch-summary').textContent=`Prompt processing: logical batch ${s.batch_size??'engine default'} · microbatch ${s.ubatch_size??'engine default'}.`;
 $('reuse-summary').textContent=`Engine defaults: host cache ${featureState?.cache_ram_mib?.advertised_default??'unknown'} MiB; checkpoints ${featureState?.context_checkpoints?.advertised_default??'unknown'} per slot. Blank preserves these values for normal launches.`;
 $('execution-summary').textContent=`Target GPU sampling: ${s.backend_sampling?'requested':'engine default'} · concurrent CUDA streams: ${s.cuda_graph_opt}.`;
 const ready=connected&&!resolving&&validationSnapshot===snapshot()&&!pendingAction;
 let title='Start Model',status='Ready to start';
 if(!connected){title='Waiting for panel…';status='Panel connection unavailable. Last-known service state may be stale.';}
 else if(pendingAction||launchJob){title='Starting…';status=`Loading ${modelName(job.settings?.model||launch?.model)}${job.started_at?' · '+Math.max(0,Math.floor(Date.now()/1000-job.started_at))+'s elapsed':''}`;}
 else if(experiment){title='Experiment running';status='The experiment owns the engine. View its progress or cancel it in Experiments.';}
 else if(running.ready&&same){title='Running';status=`Ready · ${modelName(running.settings?.model)}`;}
 else if(resolving){title='Preparing model…';status='Checking model identity, recommended settings and engine compatibility…';}
 else if(!validationSnapshot&&!validationError&&launch?.model&&s.engine){title='Checking settings…';status='Checking the selected settings before starting.';}
 else if(running.running){title=launch?.model===running.settings?.model?'Restart with these settings':`Switch to ${modelName(launch?.model)}`;status='This replaces the running model and interrupts its current requests.';}
 else if(!launch?.model){status='Choose or download a model to get started.';}
 else if(validationError){status='Setup needs attention before starting.';}
 $('start').textContent=title;$('start').disabled=!ready||!!job.active||(running.ready&&same)||view!=='launch';
 $('launch-status').textContent=status;
 $('stop').hidden=!(running.running||launchJob);$('stop').textContent=launchJob?'Cancel start':'Stop model';$('stop').disabled=!connected||pendingAction;
 $('copy-api').hidden=!running.ready;$('copy-api').disabled=!connected;
 $('running-summary').textContent=running.running&&running.settings?`Current model: ${modelName(running.settings.model)} · ${Math.floor(running.settings.context/running.settings.slots).toLocaleString()} tokens per conversation · ${running.ready?'ready':'loading'}`:'';
 $('launch-error').textContent=actionError||running.error||(job.status==='failed'?job.error:'')||(view==='launch'?validationError:'')||'';
 $('experiment-error').textContent=view==='experiments'?(!connected?'Panel connection unavailable. Reconnect before running an experiment.':validationError||actionError||running.error||''):'';
 $('setup-actions').hidden=resolving||(!validationError&&connected);
 $('selection-note').textContent=selectionNote;
 $('launch-state').textContent=running.running&&!same?'Settings shown are for the next start. The current model is unchanged.':'';
 $('global-operation').hidden=!job.active;
 $('global-operation').replaceChildren();
 if(job.active){$('global-operation').append(document.createTextNode(`${launchJob?'Model starting':'Experiment running'} · ${job.phase||job.status} `));const link=document.createElement('a');link.href=launchJob?'#launch':'#experiments';link.textContent='View progress';$('global-operation').append(link);}
 document.querySelectorAll('.run').forEach(b=>{b.disabled=!ready||!!job.active||view!=='experiments';b.textContent=(running.running?'Stop model and run · ':'')+b.dataset.label;});
 $('cancel').disabled=!connected||!job.active||pendingAction;
 $('save-default').disabled=!ready||view!=='launch';
}
function renderRecommendations(){
 const entries=(discovered.catalog||[]).filter(m=>m.recommendation).sort((a,b)=>a.recommendation.rank-b.recommendation.rank);
 const selected=settings().model;
 const markup=entries.map(m=>{
  const installed=(discovered.models||[]).filter(x=>x.catalog_id===m.id&&x.identity_verified),r=m.recommendation;
  const chosen=installed.find(x=>x.path===selected)||installed[0];
  const unverified=discovered.models?.find(x=>x.catalog_id===m.id&&!x.identity_verified);
  const d=(statusState.downloads||[]).find(d=>d.id===m.id),active=d&&['queued','downloading'].includes(d.state);
  return `<div class="model-choice">${chosen?`<input type="radio" name="recommended-model" id="recommend-${esc(m.id)}" value="${esc(chosen.path)}" ${installed.some(x=>x.path===selected)?'checked':''}>`:''}<div>${chosen?`<label for="recommend-${esc(m.id)}">`:'<div>'}<b>${esc(r.name||m.name)}</b> <span class="pill">${esc(r.label)}</span>${chosen?'</label>':'</div>'}<small>${esc(r.description)}</small><small>${esc(r.variant)} · ${chosen?'Installed':unverified?'Installed variant · identity not verified':`${m.size_gb} GB download`}${installed.length>1?' · Multiple copies in All models':''}</small>${d?`<small>${esc(d.state)} · ${d.percent}% · ${esc(d.detail||'')}</small>`:''}</div>${!chosen&&unverified&&!active?`<button data-select-model="${esc(unverified.path)}">Choose installed variant</button>`:!chosen||active?`<button data-download="${esc(m.id)}" data-active="${!!active}">${active?'Cancel download':d?.state==='error'?'Retry download':'Download model'}</button>`:''}</div>`;
 }).join('');
 if($('recommended-list').innerHTML!==markup){const focus=$('recommended-list').contains(document.activeElement)?document.activeElement.id:null;$('recommended-list').innerHTML=markup;if(focus)document.getElementById(focus)?.focus({preventScroll:true});}
}
function renderDownloads(){
 if(!discovered.catalog)return;
 const ds=statusState.downloads||[];
 $('catalog').innerHTML=[...discovered.catalog].sort((a,b)=>(a.recommendation?.rank||999)-(b.recommendation?.rank||999)).map(m=>{
  const d=ds.find(d=>d.id===m.id),installed=discovered.models.some(x=>x.catalog_id===m.id),active=d&&['queued','downloading'].includes(d.state);
  return `<div class="download"><button data-download="${esc(m.id)}" ${installed&&!active?'disabled':''} data-active="${!!active}">${active?'Cancel':installed?'Installed':'Download'}</button><b>${esc(m.name)}</b><small style="display:block">${m.size_gb} GB · ${esc(d?`${d.state} ${d.percent}% · ${d.rate_mib_s} MiB/s · ${d.detail}`:m.repo)}</small></div>`;
 }).join('');renderRecommendations();
}
async function scan(){
 if(scanPending)return false;
 let applied=false;scanPending=true;const n=++selectionSequence;resolving=true;validationSnapshot='';defaultState();launchState();
 const before=settings();
 try{
  const data=await api('/api/discover',{});
  if(n!==selectionSequence)return;
  discovered=data;applied=true;
  $('model').replaceChildren(new Option('Choose an installed model',''),...data.models.map(m=>new Option(`${modelName(m.path)} · ${m.path}`,m.path)));
  $('engine-list').replaceChildren(...data.engines.map(e=>{const o=new Option(e.path,e.path);o.label=e.devices?.join(', ')||'No GPU device';return o;}));
  renderDownloads();
  const initial=!discoveredOnce;discoveredOnce=true;
  const running=statusState.engine?.running?statusState.engine.settings:statusState.job?.kind==='launch'&&statusState.job?.active?statusState.job.settings:null;
  if(initial&&running){loadedDefaults={settings:running,mode:'running',source:'Current model settings',notes:['Showing the current operation. No saved preferences changed.']};fill(running);await inspect();}
  else if(before.model){fill(before);await inspect();}
  else{await selectModel('');}
 }catch(e){if(n===selectionSequence)validationError='Discovery failed: '+e.message;}
 finally{scanPending=false;if(n===selectionSequence){resolving=false;defaultState();launchState();}finishNavigation();}
 return applied;
}
async function selectModel(path){
 const n=++selectionSequence;validationSequence++;resolving=true;validationSnapshot='';validationError='';selectionNote='';actionError='';loadedDefaults=null;savedExists=false;
 if(path){if(!Array.from($('model').options).some(o=>o.value===path))$('model').add(new Option(modelName(path),path));$('model').value=path;}
 defaultState();launchState();
 try{
  const data=await api('/api/launch/select',{model:path,...(engineOverride?{engine:$('engine').value,backend:$('backend').value,device:$('device').value}:{})});
  if(n!==selectionSequence)return;
  selectionNote=data.reason||'';
  if(data.settings){loadedDefaults={...data,mode:'recommended'};fill(data.settings);await inspect();}
  else{validationError=data.reason;}
  if(n===selectionSequence&&(view==='experiments'||!drafts.experiments))resetExperimentCeiling();
 }catch(e){if(n===selectionSequence)validationError=e.message;}
 finally{if(n===selectionSequence){resolving=false;defaultState();renderRecommendations();launchState();finishNavigation();}}
}
async function loadDefaults(source='built-in'){
 const n=++selectionSequence,selected=settings();validationSequence++;resolving=true;validationSnapshot='';validationError='';defaultState();launchState();
 try{
  const d=await api('/api/default/resolve',{settings:selected,source});
  if(n!==selectionSequence)return;
  loadedDefaults={...d,mode:source==='saved'?'saved':'recommended'};fill(d.settings);await inspect();
 }catch(e){if(n===selectionSequence)validationError='Settings could not be loaded: '+e.message;}
 finally{if(n===selectionSequence){resolving=false;defaultState();launchState();finishNavigation();}}
}
async function inspect(){
 const n=++validationSequence,selected=settings(),key=JSON.stringify(selected);validationSnapshot='';validationError='';featureState=null;
 $('engine-alternative').hidden=true;
 $('model-facts').textContent=(discovered.models?.find(m=>m.path===selected.model)?.metadata.context?'Checkpoint metadata context limit: '+discovered.models.find(m=>m.path===selected.model).metadata.context.toLocaleString()+' tokens. ':'')+'Usable allocation is shown above.';
 launchState();
 if(!selected.model||!selected.engine){validationError=!selected.model?'Choose or download a model.':'Choose an installed GPU-enabled llama-server under Customize settings.';launchState();return;}
 $('features').textContent='Checking support…';
 try{
  const [c,v]=await Promise.all([api('/api/capabilities',{settings:selected}),api('/api/launch/check',{settings:selected})]);
  if(n!==validationSequence||key!==snapshot())return;
  featureState=c.features;savedExists=v.saved_exists;
  const devices=c.engine.devices.filter(d=>d.startsWith(selected.backend));
  $('device').replaceChildren(...[...new Set([selected.device,...devices])].map(d=>new Option(d||'Choose device',d)));
  $('device').value=selected.device;
  $('device-hint').textContent=devices.includes(selected.device)?'':`Choose a detected ${selected.backend} device or a matching engine build.`;
  $('device-probe').hidden=devices.includes(selected.device);$('device-output').textContent=c.engine.error||c.engine.device_output||'';
  const alternative=discovered.engines?.find(e=>e.path!==selected.engine&&e.devices?.some(d=>d.startsWith(selected.backend)));
  if(!devices.length&&alternative){$('engine-alternative').hidden=false;$('use-backend-engine').textContent=`Use matching ${selected.backend} build`;$('backend-engine-path').textContent=alternative.path;$('use-backend-engine').onclick=()=>attempt(async()=>{
    if(key!==snapshot())return;
    $('engine').value=alternative.path;$('device').replaceChildren(new Option(alternative.devices.find(d=>d.startsWith(selected.backend))));engineOverride=true;await loadDefaults();
  });}
  // Methods needing setup must remain selectable so their prerequisite controls can be reached.
  for(const option of $('speculation').options)option.disabled=false;
  for(const k of ['flash','cache','effort'])$(k).disabled=false;
  validationError=v.error||'';if(v.valid)validationSnapshot=key;
  renderFeatures();defaultState();launchState();
 }catch(e){if(n===validationSequence&&key===snapshot()){validationError=e.message;$('features').textContent=e.message;launchState();}}
}
function edited(key){
 selectionSequence++;validationSequence++;resolving=false;validationSnapshot='';validationError='';actionError='';clearTimeout(editTimer);
 if(key==='cache')for(const k of optionalCache)$(k).value='';
 if(key==='drafter')$('pair_confirmed').checked=false;
 if(['engine','backend','device'].includes(key))engineOverride=true;
 defaultState();slotNote();renderFeatures();launchState();
 editTimer=setTimeout(()=>attempt(inspect),250);finishNavigation();
}
async function poll(){
 if(pollPending)return;
 pollPending=true;
 try{
  const s=await api('/api/status');statusState=s;token=s.token;connected=true;
  if(startAttempt&&s.job.request_id===startAttempt.request_id&&['failed','cancelled','serving'].includes(s.job.status))startAttempt=null;
  $('gpu').textContent=s.hardware.gpus.length?s.hardware.gpus.map(g=>`${g.name} · ${g.used_mib.toLocaleString()} / ${g.total_mib.toLocaleString()} MiB`).join(' | '):'No NVIDIA GPU detected';
  $('endpoint').textContent=s.engine.ready?`API on model workstation: ${s.endpoint}`:'';
  $('paths').textContent=`Models: ${s.paths.models}. Engine roots: ${s.paths.engines.join(', ')}. Configure LLLM2_MODELS_DIR and LLLM2_ENGINE_ROOTS before starting the panel.`;
  $('job').textContent=`${s.job.status}${s.job.current?' · '+s.job.current:''}`;
  $('job-detail').textContent=[s.job.phase,s.job.error,...(s.job.skipped||[]).map(x=>`${x.option}: ${x.reason}`)].filter(Boolean).join(' · ');
  $('bar').style.width=s.job.total?`${100*(s.job.completed||0)/s.job.total}%`:'0%';
  $('logs').textContent=[(s.engine.argv||[]).join(' '),'',...(s.engine.logs||[])].join('\n');
  renderDownloads();launchState();
  const complete=(s.downloads||[]).filter(d=>d.state==='complete').map(d=>d.id).sort().join('|');
  if(complete!==lastDownloads){if(!discoveredOnce)lastDownloads=complete;else if(!scanPending&&await scan())lastDownloads=complete;}
  if(view==='experiments'){try{await refreshResults();}catch(e){message('Comparisons could not be refreshed: '+e.message,true);}}
 }catch(e){connected=false;launchState();}
 finally{pollPending=false;}
}
$('refresh').onclick=()=>attempt(scan);
$('retry-setup').onclick=()=>attempt(async()=>{await poll();await scan();});
$('choose-engine').onclick=()=>{$('customize').open=true;$('engine').focus();};
$('model').onchange=()=>attempt(()=>selectModel($('model').value));
$('recommended-list').onchange=e=>{if(e.target.matches('input[type=radio]'))attempt(()=>selectModel(e.target.value));};
for(const k of settingKeys){if(k!=='model')$(k).addEventListener('input',()=>edited(k));}
for(const k of ['engine','backend'])$(k).onchange=()=>{clearTimeout(editTimer);attempt(loadDefaults);};
$('load-default').onclick=()=>attempt(()=>loadDefaults('saved'));
$('built-in-default').onclick=()=>attempt(()=>loadDefaults('built-in'));
$('save-default').onclick=()=>attempt(async()=>{
 const selected=settings(),key=snapshot(),savedView=view,provenance=loadedDefaults?.mode==='result'&&sameSettings(selected,loadedDefaults.settings)?loadedDefaults:null;
 const saved=await api('/api/default/save',provenance?{result_id:provenance.result_id,use_context:provenance.use_context}:{settings:selected});
 if(key===snapshot()&&savedView===view){loadedDefaults={settings:saved,mode:'saved',source:provenance?'My saved settings · experiment evidence':'My saved settings · manual preferences',notes:provenance?.notes||['Manual preferences are not a measured benefit.'],evidence:provenance?.evidence};savedExists=true;defaultState();}
 message('Settings saved for this model and backend.');
});
$('start').onclick=()=>attempt(async()=>{
 if($('start').disabled||view!=='launch')return;
 const selected=structuredClone(settings()),running=statusState.engine||{};
 if(!startAttempt||!sameSettings(startAttempt.settings,selected))startAttempt={request_id:crypto.randomUUID?crypto.randomUUID():`${Date.now()}-${Math.random()}`,settings:selected};
 pendingAction=true;actionError='';launchState();
 try{await api('/api/start',{...startAttempt,replace_running:!!running.running,expected_pid:running.pid});startAttempt=null;}
 catch(e){actionError='Start could not be confirmed: '+e.message+'. Check service status before retrying.';}
 finally{await poll();pendingAction=false;launchState();}
});
for(const k of ['stop','cancel'])$(k).onclick=()=>attempt(async()=>{
 if(pendingAction)return;pendingAction=true;launchState();
 try{await api('/api/'+k,{});actionError='';}finally{await poll();pendingAction=false;launchState();}
});
$('copy-api').onclick=()=>attempt(async()=>{
 try{await navigator.clipboard.writeText(statusState.endpoint);message('API address copied. This loopback address is on the model workstation.');}
 catch{message('API on the model workstation: '+statusState.endpoint);}
});
async function downloadClick(e){const variant=e.target.closest('[data-select-model]');if(variant){await selectModel(variant.dataset.selectModel);return;}const b=e.target.closest('[data-download]');if(!b||b.disabled)return;b.disabled=true;try{await api(b.dataset.active==='true'?'/api/download/cancel':'/api/download',{id:b.dataset.download});await poll();}finally{renderDownloads();}}
for(const id of ['catalog','recommended-list'])$(id).onclick=e=>attempt(()=>downloadClick(e));
$('use-launch-settings').onclick=()=>attempt(async()=>{
 if(view!=='experiments')return;
 selectionSequence++;clearTimeout(editTimer);resolving=false;const d=structuredClone(drafts.launch);loadedDefaults=d.defaults;engineOverride=d.engineOverride;fill(d.settings);resetExperimentCeiling();await inspect();
});
document.querySelectorAll('.run').forEach(b=>{
 b.dataset.label=b.textContent;
 b.onclick=()=>attempt(async()=>{
  if(b.disabled||view!=='experiments')return;
  const data={settings:structuredClone(settings()),mode:b.dataset.mode,combinations:structuredClone(combinations),replace_running:!!statusState.engine?.running,expected_pid:statusState.engine?.pid,workloads:[...document.querySelectorAll('#workloads input:checked')].map(x=>x.value),search_context:$('search_context').checked,sweep_prompts:$('sweep_prompts').checked,full_window:$('full_window').checked};
  for(const k of ['prompt_tokens','output_tokens','timeout','context_timeout','repeats','max_context'])data[k]=Number($(k).value);
  if(data.mode==='warm-conversation'){data.search_context=false;data.sweep_prompts=false;data.full_window=false;data.workloads=['long-code'];}
  pendingAction=true;launchState();
  try{await api('/api/benchmark',data);message('Experiment queued. Partial results are saved.');}finally{await poll();pendingAction=false;launchState();}
 });
});
$('add-combo').onclick=()=>{combinations.push(structuredClone(settings()));$('combos').textContent=combinations.map((s,i)=>`${i+1}. ${modelName(s.model)} · ${s.speculation} / ${cacheLabel(s)} / flash ${s.flash} / effort ${s.effort} / draft ${s.draft_length}`).join('\n');};
$('clear-combos').onclick=()=>{combinations=[];$('combos').textContent='No combinations selected.';};
$('results').onclick=e=>{const b=e.target.closest('[data-promote],[data-context]');if(b)attempt(async()=>{
 const n=++selectionSequence;validationSequence++;resolving=true;validationSnapshot='';launchState();
 try{
  const d=await api('/api/result/preview',{result_id:b.dataset.promote||b.dataset.context,use_context:!!b.dataset.context});
  if(n!==selectionSequence)return;
  const expected=n+(view==='launch'?0:1);resolving=false;await switchView('launch');
  if(selectionSequence!==expected)return;
  location.hash='launch';loadedDefaults={...d,mode:'result'};fill(d.settings);await inspect();
  message('Experiment settings copied to Launch. Review before starting or saving; your saved settings are unchanged.');
 }finally{if(n===selectionSequence){resolving=false;defaultState();launchState();finishNavigation();}}
});};
(async()=>{await poll();await scan();if(location.hash==='#experiments')await switchView('experiments');slotNote();setInterval(poll,2500);})();
