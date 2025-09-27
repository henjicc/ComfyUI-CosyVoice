import { app } from "../../../scripts/app.js";

// 显示保存说话人信息的节点
app.registerExtension({
    name: "ComfyUI-CosyVoice.SaveSpeakerNode",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "CosyVoice2SaveSpeaker") {
            // 添加多行文本显示框
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);
                
                if (message.text) {
                    // 创建显示区域
                    const widget = this.widgets.find(w => w.name === "save_info");
                    if (widget) {
                        widget.value = message.text[0];
                    } else {
                        // 创建新的文本显示widget
                        const textBox = this.addDOMWidget("save_info", "customtext", 
                            document.createElement("div"), {
                                getValue() {
                                    return this.element.textContent;
                                },
                                setValue(v) {
                                    this.element.textContent = v;
                                }
                            });
                        
                        // 设置样式
                        textBox.element.style.background = "#222";
                        textBox.element.style.color = "#31EC88";
                        textBox.element.style.padding = "10px";
                        textBox.element.style.borderRadius = "5px";
                        textBox.element.style.whiteSpace = "pre-wrap";
                        textBox.element.style.overflow = "auto";
                        textBox.element.style.maxHeight = "150px";
                        textBox.element.style.fontSize = "12px";
                        textBox.element.textContent = message.text[0];
                        
                        // 设置widget属性
                        textBox.serializeValue = () => {
                            return textBox.element.textContent;
                        };
                    }
                    
                    // 触发节点重绘
                    this.setSize(this.computeSize());
                }
            };
        }
    }
});